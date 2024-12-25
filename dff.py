
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
    output_from_target_layer = torch.from_numpy(activations).to(device)
    print('activation_shape:', activations.shape)

    with torch.no_grad():
        for layer in model.model.model.model[target_layer_index + 2:]:
            output_from_target_layer = layer(output_from_target_layer) 
    print('output_shape:', output_from_target_layer.shape)
    reshaped_activations = activations.transpose((1, 0, 2, 3))
    reshaped_activations[np.isnan(reshaped_activations)] = 0
    reshaped_activations = reshaped_activations.reshape(
        reshaped_activations.shape[0], -1)
    offset = reshaped_activations.min(axis=-1)
    reshaped_activations = reshaped_activations - offset[:, None]
    print('reshaped activations', activations.shape)

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
# Load YOLOv5 model and move it to the appropriate device
img, rgb_img_float, input_tensor = get_image_from_url("/home/drovco/Bhumika/NeuralVista/data/xai/sample1.jpeg", device)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
dff= DeepFeatureFactorization(model=model, 
                               target_layer=model.model.model.model[-4], 
                               computation_on_concepts=None)
n_components = 4
concepts, batch_explanations = dff(input_tensor, model, n_components)

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
