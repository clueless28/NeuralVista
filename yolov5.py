
import torch
import cv2
import numpy as np
from PIL import Image  
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import gradio as gr
import os
from typing import Callable, List, Tuple, Optional
from sklearn.decomposition import NMF
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, create_labels_legend, show_factorization_on_image
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_factorization_on_image
import requests    
import yaml
import matplotlib.patches as patches



# Global Color Palette
COLORS = np.random.uniform(0, 255, size=(80, 3))


def parse_detections(results):
    detections = results.pandas().xyxy[0].to_dict()
    boxes, colors, names, classes = [], [], [], []
    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin, ymin = int(detections["xmin"][i]), int(detections["ymin"][i])
        xmax, ymax = int(detections["xmax"][i]), int(detections["ymax"][i])
        name, category = detections["name"][i], int(detections["class"][i])
        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(COLORS[category])
        names.append(name)
        classes.append(category) 
    return boxes, colors, names, classes


def draw_detections(boxes, colors, names, classes, img):
    for box, color, name, cls in zip(boxes, colors, names, classes):
        xmin, ymin, xmax, ymax = box
        label = f"{cls}: {name}"  # Combine class ID and name
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            img, label, (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            lineType=cv2.LINE_AA
        )
    return img


def generate_cam_image(model, target_layers, tensor, rgb_img, boxes):
    cam = EigenCAM(model, target_layers)
    grayscale_cam = cam(tensor)[0, :, :]
    img_float = np.float32(rgb_img) / 255
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    renormalized_cam_image = show_cam_on_image(img_float, renormalized_cam, use_rgb=True)

    return cam_image, renormalized_cam_image


def xai_yolov5(image,target_lyr = -5, n_components = 8):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    model.cpu()

    target_layers = [model.model.model.model[-2]]  # Grad-CAM target layer

    # Run YOLO detection
    results = model([image])
    boxes, colors, names, classes = parse_detections(results) 
    detections_img = draw_detections(boxes, colors, names,classes,  image.copy())

    # Prepare input tensor for Grad-CAM
    img_float = np.float32(image) / 255
    transform = transforms.ToTensor()
    tensor = transform(img_float).unsqueeze(0)

    # Grad-CAM visualization
    cam_image, renormalized_cam_image = generate_cam_image(model, target_layers, tensor, image, boxes)

    rgb_img_float, batch_explanations, result = dff_nmf(image, target_lyr = -5, n_components = 8)
    #result = np.hstack(result)
    im = visualize_batch_explanations(rgb_img_float, batch_explanations)  ##########to be displayed
    
    # Combine results
    final_image = np.hstack((image, detections_img, renormalized_cam_image))
    caption = "Results using YOLOv5"
    return Image.fromarray(final_image), caption, result


def dff_l(activations,  model,  n_components):
    batch_size, channels, h, w = activations.shape
    print('activation', activations.shape)
    target_layer_index = 4        
    reshaped_activations = activations.transpose((1, 0, 2, 3))
    reshaped_activations[np.isnan(reshaped_activations)] = 0
    reshaped_activations = reshaped_activations.reshape(
        reshaped_activations.shape[0], -1)
    offset = reshaped_activations.min(axis=-1)
    reshaped_activations = reshaped_activations - offset[:, None]
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(reshaped_activations)
    H = model.components_
    concepts = W + offset[:, None]
    explanations = H.reshape(n_components, batch_size, h, w)
    explanations = explanations.transpose((1, 0, 2, 3))
    return concepts, explanations

class DeepFeatureFactorization:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 reshape_transform: Callable = None,
                 computation_on_concepts=None
                 ):
        self.model = model
        self.computation_on_concepts = computation_on_concepts
        self.activations_and_grads = ActivationsAndGradients(
            self.model, [target_layer], reshape_transform)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 model: torch.nn.Module,
                 n_components: int = 16):
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor) 

        batch_size, channels, h, w = input_tensor.size()
        _ = self.activations_and_grads(input_tensor)

        with torch.no_grad():
            activations = self.activations_and_grads.activations[0].cpu(
            ).numpy()

        concepts, explanations = dff_l(activations, model,  n_components=n_components)
        processed_explanations = []

        for batch in explanations:
            processed_explanations.append(scale_cam_image(batch, (w, h)))

        if self.computation_on_concepts:
            with torch.no_grad():
                concept_tensors = torch.from_numpy(
                    np.float32(concepts).transpose((1, 0)))
                concept_outputs = self.computation_on_concepts(
                    concept_tensors).cpu().numpy()
            return concepts, processed_explanations, concept_outputs
        else:
            return concepts, processed_explanations,  explanations

    def __del__(self):
        self.activations_and_grads.release()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in ActivationSummary with block: {exc_type}. Message: {exc_value}")
            return True


def dff_nmf(image, target_lyr, n_components):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]  # Mean for RGB channels
    std = [0.229, 0.224, 0.225]   # Standard deviation for RGB channels
    img = cv2.resize(image, (640, 640))
    rgb_img_float = np.float32(img) / 255.0
    input_tensor = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).to(device) 

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    dff= DeepFeatureFactorization(model=model,    
                               target_layer=model.model.model.model[int(target_lyr)],    
                               computation_on_concepts=None)
    
    concepts, batch_explanations, explanations = dff(input_tensor, model, n_components)


    yolov5_categories_url = \
            "https://github.com/ultralytics/yolov5/raw/master/data/coco128.yaml"  # URL to the YOLOv5 categories file
    yaml_data = requests.get(yolov5_categories_url).text
    labels = yaml.safe_load(yaml_data)['names']  # Parse the YAML file to get class names
    num_classes = model.model.model.model[-1].nc 
    results = []
    for indx in range(explanations[0].shape[0]):
        upsampled_input =  explanations[0][indx]
        upsampled_input = torch.tensor(upsampled_input)
        device = next(model.parameters()).device
        input_tensor = upsampled_input.unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(1).repeat(1, 128, 1, 1)    
        detection_lyr = model.model.model.model[-1]
        output1 = detection_lyr.m[0](input_tensor.to(device))
        objectness = output1[..., 4]  # Objectness score (index 4)
        class_scores = output1[..., 5:]  # Class scores (from index 5 onwards, representing 80 classes)
        objectness = torch.sigmoid(objectness)
        class_scores = torch.sigmoid(class_scores)
        confidence_mask = objectness > 0.5
        objectness = objectness[confidence_mask]
        class_scores = class_scores[confidence_mask]
        scores, class_ids = class_scores.max(dim=-1)  # Get max class score per cell
        scores = scores * objectness  # Adjust scores by objectness
        boxes = output1[..., :4]  # First 4 values are x1, y1, x2, y2
        boxes = boxes[confidence_mask]  # Filter boxes by confidence mask
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.axis("off")
        ax.imshow(torch.tensor(batch_explanations[0][indx]).cpu().numpy(), cmap="plasma")  # Display image
        top_score_idx = scores.argmax(dim=0)  # Get the index of the max score
        top_score = scores[top_score_idx].item()
        top_class_id = class_ids[top_score_idx].item()
        top_box = boxes[top_score_idx].cpu().numpy()
        scale_factor = 16
        x1, y1, x2, y2 = top_box
        x1, y1, x2, y2 = x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor
        rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
        predicted_label = labels[top_class_id]  # Map ID to label
        ax.text(x1, y1, f"{predicted_label}: {top_score:.2f}",
            color='r', fontsize=12, verticalalignment='top')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()  # Draw the canvas to make sure the image is rendered
        image_array = np.array(fig.canvas.renderer.buffer_rgba())  # Convert to numpy array
        print("____________image_arrya", image_array.shape)
        image_resized = cv2.resize(image_array, (640, 640))
        rgba_channels = cv2.split(image_resized)
        alpha_channel = rgba_channels[3] 
        rgb_channels = np.stack(rgba_channels[:3], axis=-1)
        #overlay_img = (alpha_channel[..., None] * image) + ((1 - alpha_channel[..., None]) * rgb_channels)
        
        #temp = image_array.reshape((rgb_img_float.shape[0],rgb_img_float.shape[1]) )
        #visualization = show_factorization_on_image(rgb_img_float, image_array.resize((rgb_img_float.shape)) , image_weight=0.3)
        visualization = show_factorization_on_image(rgb_img_float, np.transpose(rgb_channels, (2, 0, 1)), image_weight=0.3)
        results.append(visualization)
        plt.clf()  
        #return image_array


    return rgb_img_float, batch_explanations, results


def visualize_batch_explanations(rgb_img_float, batch_explanations, image_weight=0.7):
    for i, explanation in enumerate(batch_explanations):
        # Create visualization for each explanation
        print("visualization concepts",rgb_img_float.shape,explanation.shape  )
        visualization = show_factorization_on_image(rgb_img_float, explanation, image_weight=image_weight)
        plt.figure()
        plt.imshow(visualization)  # Correctly pass the visualization data
        plt.title(f'Explanation {i + 1}')  # Set the title for each plot
        plt.axis('off')  # Hide axes
        plt.show()  # Show the plot
    plt.savefig("test_w.png")
    print('viz', visualization.shape)
    return visualization
