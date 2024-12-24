import yaml
import torch
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import numpy as np
import requests
import cv2
import os
import torch
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.image import deprocess_image, show_factorization_on_image
"""
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]  # Mean for RGB channels
std = [0.229, 0.224, 0.225]   # Standard deviation for RGB channels
# Load YOLOv5 model and move it to the appropriate device
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
print(f"Loaded YOLOv5 model on {device}")

def create_labels(concept_scores, top_k=2):
    
    yolov5_categories_url = \
        "https://github.com/ultralytics/yolov5/raw/master/data/coco128.yaml"  # URL to the YOLOv5 categories file
    yaml_data = requests.get(yolov5_categories_url).text
    labels = yaml.safe_load(yaml_data)['names']  # Parse the YAML file to get class names
    print(concept_scores)
    print(labels)
    
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]    
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk
def get_image_from_url(url, device):
 

    img = np.array(Image.open(os.path.join(os.getcwd(), "data/xai/sample1.jpeg")))
    img = cv2.resize(img, (640, 640))
    rgb_img_float = np.float32(img) /255.0
    input_tensor = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return img, rgb_img_float, input_tensor

def visualize_image(model, img_url, n_components=10, top_k=1, lyr_idx = 2):
    img, rgb_img_float, input_tensor = get_image_from_url(img_url, device)
    
    # Specify the target layer for DeepFeatureFactorization (e.g., YOLO's backbone)
    target_layer = model.model.model.model[-lyr_idx]  # Select a feature extraction layer
    
    dff = DeepFeatureFactorization(model=model.model, target_layer=target_layer)
    
    # Run DFF on the input tensor
    concepts, batch_explanations = dff(input_tensor, n_components)
    
    # Softmax normalization
    concept_outputs = torch.softmax(torch.from_numpy(concepts), axis=-1).numpy()    
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    
    # Visualize explanations
    visualization = show_factorization_on_image(rgb_img_float, 
                                                batch_explanations[0],
                                                image_weight=0.2,
                                                concept_labels=concept_label_strings)
    
    import matplotlib.pyplot as plt
    plt.imshow(visualization)
    plt.savefig("test" + str(lyr_idx) + ".png")
    result = np.hstack((img, visualization))

    
    # Resize for visualization
    if result.shape[0] > 500:
        result = cv2.resize(result, (result.shape[1]//4, result.shape[0]//4))
    
    return result

# Test with images

Image.fromarray(visualize_image(model, 
                                        "https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/both.png?raw=true", lyr_idx = 2))

                                        
"""
"""
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]  # Mean for RGB channels
std = [0.229, 0.224, 0.225]   # Standard deviation for RGB channels
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
from sklearn.decomposition import NMF

print(f"Loaded YOLOv5 model on {device}")

def create_labels(concept_scores, top_k=2):
    
    yolov5_categories_url = \
        "https://github.com/ultralytics/yolov5/raw/master/data/coco128.yaml"  # URL to the YOLOv5 categories file
    yaml_data = requests.get(yolov5_categories_url).text
    labels = yaml.safe_load(yaml_data)['names']  # Parse the YAML file to get class names
    print(concept_scores)
    print(labels)
    
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]    
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk

def get_image_from_url(url, device):
    img = np.array(Image.open(os.path.join(os.getcwd(), "data/xai/sample1.jpeg")))
    img = cv2.resize(img, (640, 640))
    rgb_img_float = np.float32(img) /255.0
    input_tensor = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return img, rgb_img_float, input_tensor

def visualize_image(model, img_url, n_components=10, top_k=1, lyr_idx = 2):
    img, rgb_img_float, input_tensor = get_image_from_url(img_url, device)
    
    # Specify the target layer for DeepFeatureFactorization (e.g., YOLO's backbone)
    target_layer = model.model.model.model[-lyr_idx]  # Select a feature extraction layer
    for idx, layer in enumerate(model.model.model.model):
        print(f"Layer {idx}: {layer}")
    feature_maps = []
    
    # Hook function to capture output of target layer, register a forward hook on the target layer from which you want to extract features.
    def hook(module, input, output):
        feature_maps.append(output)
    target_layer.register_forward_hook(hook)

    with torch.no_grad():
        outputs = model(input_tensor)
    if feature_maps:
        feature_map = feature_maps[0]
        feature_map = feature_map.squeeze(0)
        n_channels = feature_map.shape[0]
        n_components = min(n_components, n_channels)
        fig, axes = plt.subplots(1, n_components, figsize=(15, 5))
        for i in range(n_components):
            ax = axes[i]
            ax.imshow(feature_map[i].cpu().numpy(), cmap='viridis')  # Use a colormap for better visualization
            ax.axis('off')
            ax.set_title(f'Channel {i+1}')

        plt.tight_layout()
        plt.show()
        plt.savefig("feature_map" + str(lyr_idx) + ".png")

    if feature_maps:
        feature_map = feature_maps[0].squeeze(0) 
        n_channels = feature_map.shape[0]

        feature_map_reshaped = feature_map.view(n_channels, -1).cpu().numpy() 

        # Check for negative values
        if np.any(feature_map_reshaped < 0):
            print("Negative values found in feature map. Clipping to zero.")
            feature_map_reshaped = np.clip(feature_map_reshaped, a_min=0, a_max=None)  # Clip negative values

        # Apply Non-Negative Matrix Factorization
        n_components = 10  # Number of features to extract
        nmf_model = NMF(n_components=n_components, init='random', random_state=0)
        
        W = nmf_model.fit_transform(feature_map_reshaped)  # Feature matrix
        H = nmf_model.components_  # Coefficient matrix
        print('factorization matrix', W.shape, H.shape)

        # Visualize the basis features (W)
        fig, axes = plt.subplots(1, n_components, figsize=(15, 5))
        for i in range(n_components):
            ax = axes[i]
            ax.imshow(H[i].reshape(feature_map.shape[1], feature_map.shape[2]), cmap='viridis')  # Reshape to spatial dimensions
            ax.axis('off')
            ax.set_title(f'Feature {i+1}')

        plt.tight_layout()
        plt.show()
        plt.savefig("test1.png")  # Save before displaying




# Test with images

Image.fromarray(visualize_image(model, 
                                        "https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/both.png?raw=true", lyr_idx = 2))
"""
import numpy as np
from PIL import Image
import torch
from typing import Callable, List, Tuple, Optional
from sklearn.decomposition import NMF
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, create_labels_legend, show_factorization_on_image

def dff_l(activations: np.ndarray, n_components: int = 5):
    """ Compute Deep Feature Factorization on a 2d Activations tensor.

    :param activations: A numpy array of shape batch x channels x height x width
    :param n_components: The number of components for the non negative matrix factorization
    :returns: A tuple of the concepts (a numpy array with shape channels x components),
              and the explanation heatmaps (a numpy arary with shape batch x height x width)

    W is the feature matrix, representing extracted features or components.
    H is the coefficient matrix, representing how these features combine to reconstruct each sample in the dataset.
    """

    batch_size, channels, h, w = activations.shape
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
                 n_components: int = 16):
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor)  # Convert NumPy array 

        print("Input shape:", input_tensor.shape)
        batch_size, channels, h, w = input_tensor.size()
        _ = self.activations_and_grads(input_tensor)

        with torch.no_grad():
            activations = self.activations_and_grads.activations[0].cpu(
            ).numpy()

        concepts, explanations = dff_l(activations, n_components=n_components)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]  # Mean for RGB channels
std = [0.229, 0.224, 0.225]   # Standard deviation for RGB channels
# Load YOLOv5 model and move it to the appropriate device
img, rgb_img_float, input_tensor = get_image_from_url("/home/drovco/Bhumika/NeuralVista/data/xai/sample1.jpeg", device)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
dff = DeepFeatureFactorization(model=model, 
                               target_layer=model.model.model.model[-2], 
                               computation_on_concepts=None)
n_components = 30
concepts, batch_explanations = dff(input_tensor, n_components=n_components)
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_factorization_on_image

def visualize_batch_explanations(rgb_img_float, batch_explanations, image_weight=0.3):
    # Iterate over each explanation in the batch
    for i, explanation in enumerate(batch_explanations):
        # Create visualization for each explanation
        visualization = show_factorization_on_image(rgb_img_float, explanation, image_weight=image_weight)
        
        # Display the visualization using Matplotlib
        plt.figure()
        plt.imshow(visualization)  # Correctly pass the visualization data
        plt.title(f'Explanation {i + 1}')  # Set the title for each plot
        plt.axis('off')  # Hide axes
        plt.show()  # Show the plot
    plt.savefig("test_w.png")

# Assuming rgb_img_float and batch_explanations are defined as per your previous code
visualize_batch_explanations(rgb_img_float, batch_explanations)
