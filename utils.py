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
import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, n_components, figsize=(20, 5))
    for i in range(n_components):
        axs[i].imshow(batch_explanations[0][i])
        axs[i].title.set_text('Concept ' + str(i+1))
        axs[i].axis('off')
    plt.show()
    plt.savefig("batch_exp.png")

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
