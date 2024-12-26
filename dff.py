
import numpy as np
from PIL import Image
import torch
import cv2
from typing import Callable, List, Tuple, Optional
from sklearn.decomposition import NMF
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, create_labels_legend, show_factorization_on_image
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_factorization_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def dff_l(activations,  model,  n_components):
    batch_size, channels, h, w = activations.shape
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
            input_tensor = torch.from_numpy(input_tensor)  # Convert NumPy array 

        print("Input shape:", input_tensor.shape)
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
            return concepts, processed_explanations

    def __del__(self):
        self.activations_and_grads.release()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in ActivationSummary with block: {exc_type}. Message: {exc_value}")
            return True

def get_image_from_url(url, device):
    img = np.array(Image.open(url))  # Load image directly from URL
    img = cv2.resize(img, (640, 640))  # Resize image
    rgb_img_float = np.float32(img) / 255.0  # Normalize to [0, 1]
    input_tensor = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).to(device)  # Change shape to [1, C, H, W]
    return img, rgb_img_float, input_tensor


mean = [0.485, 0.456, 0.406]  # Mean for RGB channels
std = [0.229, 0.224, 0.225]   # Standard deviation for RGB channels
img, rgb_img_float, input_tensor = get_image_from_url("/home/drovco/Bhumika/NeuralVista/data/xai/sample1.jpeg", device)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
dff= DeepFeatureFactorization(model=model,    
                               target_layer=model.model.model.model[-5],    
                               computation_on_concepts=None)
n_components = 4
concepts, batch_explanations = dff(input_tensor, model, n_components)

print('concepts', concepts.shape)

for indx in range(batch_explanations[0].shape[0]):
    upsampled_input = batch_explanations[0][indx]
    device = next(model.parameters()).device  # Get the device of the model

    # Move input tensor to the same device
    upsampled_input = torch.from_numpy(upsampled_input)
    input_tensor = upsampled_input.unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(1).repeat(1, 128, 1, 1)

    # Now, forward pass
    detection_lyr = model.model.model.model[-1]


   # output = detection_lyr(input_tensor)
    output1 = detection_lyr.m[0](input_tensor.to(device))
    #output2 = detection_lyr.m[1](output1)
    #output3 = detection_lyr.m[2](output2)
   # print(output1.shape)  # [1, 255, 640, 640]
    x = output1[..., 0]  # x center
    y = output1[..., 1]  # y center
    w = output1[..., 2]  # width
    h = output1[..., 3]  # height
    objectness = output1[..., 4]  # objectness score
    class_scores = output1[..., 5:]  # class scores (softmax output)

    # Apply sigmoid to x, y, w, h, objectness and class scores
    x = torch.sigmoid(x)
    print('x', x.shape)
    y = torch.sigmoid(y)
    objectness = torch.sigmoid(objectness)
    class_scores = torch.sigmoid(class_scores)

    # 2. Apply the confidence threshold (filter boxes with low objectness score)
    confidence_mask = objectness > 0.5
    x = x[confidence_mask]
    y = y[confidence_mask]
    w = w[confidence_mask]
    h = h[confidence_mask]
    objectness = objectness[confidence_mask]
    class_scores = class_scores[confidence_mask]

    # 3. Convert to absolute coordinates (scale with grid size)
    batch_size, num_preds, grid_h, grid_w = output1.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(grid_w), torch.arange(grid_h))  # Get grid
    grid_x = grid_x.to(device).float()
    grid_y = grid_y.to(device).float()
    

    # Adjust the boxes relative to the grid
   #x = (x + grid_x) * (640 / grid_w)  # Scale to image size (640 is the input size)
   # y = (y + grid_y) * (640 / grid_h)
   # w = torch.exp(w) * 640 / grid_w  # scale the width (exponent for ratio)
   # h = torch.exp(h) * 640 / grid_h  # scale the height (exponent for ratio)

    # 4. Get class predictions (take the class with the highest score)
    scores, classes = class_scores.max(dim=-1)
    scores = scores * objectness  # Adjust scores by objectness
    class_ids = classes

    NMS_THRESHOLD = 0.4
    from torchvision.ops import nms
    # 5. Perform NMS to remove overlapping boxes
    boxes = torch.stack([x, y, w, h], dim=-1)  # [x, y, w, h]
    indices = nms(boxes, scores, NMS_THRESHOLD)

    # 6. Visualize the predictions (showing the boxes and class labels)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    #img = np.random.rand(640, 640, 3)  # Placeholder image (replace with your actual image)

    # Visualize predictions
    for idx in indices:
        box = boxes[idx].cpu().numpy()
        score = scores[idx].cpu().numpy()
        class_id = class_ids[idx].cpu().numpy()

        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h

        # Draw the bounding box
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, color='red', linewidth=2))
        
        # Annotate with class name and score
        ax.text(x1, y1, f'{class_id}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    #ax.imshow(img)
    ax.set_title('Predictions')
    plt.show()
    plt.savefig("output1.png")




















   # print(output2.shape)  # [1, 255, 640, 640]
   # print(output3.shape)  # [1, 255, 640, 640]

   # print(output)
  #  plt.savefig("test_" + str(indx) + ".png")
    
   # with torch.no_grad():
      #  detections = model(upsampled_input)
    #results = detections.xyxy[0] 
   # print(results)
    #for *box, conf, cls in results:
       # print(f'Box: {box}, Confidence: {conf}, Class: {cls}')


"""
    import torch.nn.functional as F
    activations_tensor = torch.from_numpy(activations)
    upsampled_input = F.interpolate(activations_tensor, size=(640, 640), mode='bilinear', align_corners=False)
    print('upsampled_input', upsampled_input.shape)
    with torch.no_grad():
        detections = model(upsampled_input)
    results = detections.xyxy[0]  # Get results in xyxy format
    # Print results
    for *box, conf, cls in results:
        print(f'Box: {box}, Confidence: {conf}, Class: {cls}')

"""


def visualize_batch_explanations(rgb_img_float, batch_explanations, image_weight=0.7):
    for i, explanation in enumerate(batch_explanations):
        # Create visualization for each explanation
        visualization = show_factorization_on_image(rgb_img_float, explanation, image_weight=image_weight)
        plt.figure()
        plt.imshow(visualization)  # Correctly pass the visualization data
        plt.title(f'Explanation {i + 1}')  # Set the title for each plot
        plt.axis('off')  # Hide axes
        plt.show()  # Show the plot
    plt.savefig("test_w.png")

visualize_batch_explanations(rgb_img_float, batch_explanations)

