import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import gradio as gr
from ultralytics import YOLO
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



COLORS = np.random.uniform(0, 255, size=(80, 3))
def parse_detections(detections, model):
    boxes, colors, names, classes = [], [], [], []
    for detection in detections.boxes:
        xmin, ymin, xmax, ymax = map(int, detection.xyxy[0].tolist())
        confidence = detection.conf.item()
        if confidence < 0.2:
            continue
        class_id = int(detection.cls.item())
        name = model.names[class_id]
        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(COLORS[class_id])
        names.append(name)
        classes.append(class_id) 
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
    model_output = model(tensor)[0]  # Adjust based on output structure
    grayscale_cam = cam(tensor, targets=model_output)[0, :, :]
    img_float = np.float32(rgb_img) / 255
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    renormalized_cam_image = show_cam_on_image(img_float, renormalized_cam, use_rgb=True)

    return cam_image, renormalized_cam_image

def xai_yolov8s(image):
    model = YOLO('yolov8s.pt')  # Ensure the model weights are available
    model.eval()
    results = model(image)
    detections = results[0]
    boxes, colors, names, classes = parse_detections(detections, model)
    detections_img = draw_detections(boxes, colors, names, classes, image.copy())
    img_float = np.float32(image) / 255
    transform = transforms.ToTensor()
    tensor = transform(img_float).unsqueeze(0)
    target_layers = [model.model.model[-2]]  # Adjust to YOLOv8 architecture
    cam_image, renormalized_cam_image = generate_cam_image(model.model, target_layers, tensor, image, boxes)
    rgb_img_float, batch_explanations, result = dff_nmf(image, target_lyr = -5, n_components = 8)
    
    final_image = np.hstack((image, detections_img, renormalized_cam_image))
    caption = "Results using YOLOv8"
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

    model = YOLO('yolov8s.pt')  # Ensure the model is loaded correctly
    dff = DeepFeatureFactorization(model=model,
                                   target_layer=model.model.model[int(target_lyr)],
                                   computation_on_concepts=None)

    concepts, batch_explanations, explanations = dff(input_tensor, model, n_components)
    results = []
    for indx in range(explanations[0].shape[0]):
        upsampled_input =  explanations[0][indx]
        upsampled_input = torch.tensor(upsampled_input)
        device = next(model.parameters()).device
        input_tensor = upsampled_input.unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(1).repeat(1, 128, 1, 1)   
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.axis("off")
        ax.imshow(torch.tensor(batch_explanations[0][indx]).cpu().numpy(), cmap="plasma")  # Display i
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
