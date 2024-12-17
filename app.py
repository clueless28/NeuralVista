import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import gradio as gr

# Global Color Palette
COLORS = np.random.uniform(0, 255, size=(80, 3))

# Function to parse YOLO detections
def parse_detections(results):
    detections = results.pandas().xyxy[0].to_dict()
    boxes, colors, names = [], [], []
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
    return boxes, colors, names

# Draw bounding boxes and labels
def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img

# Load the appropriate YOLO model based on the version
def load_yolo_model(version="yolov5"):
    if version == "yolov3":
        model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)
    elif version == "yolov5":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    elif version == "yolov7":
        model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)
    elif version == "yolov8":
        model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5', pretrained=True)  # YOLOv8 is part of the yolov5 repo starting from v7.0
    elif version == "yolov10":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)  # Placeholder for YOLOv10 (use an appropriate version if available)
    else:
        raise ValueError(f"Unsupported YOLO version: {version}")
    
    model.eval()  # Set to evaluation mode
    model.cpu()
    return model

# Main function for Grad-CAM visualization
# Main function for Grad-CAM visualization
def process_image(image, yolo_versions=["yolov5"]):
    image = np.array(image)
    image = cv2.resize(image, (640, 640))
    rgb_img = image.copy()
    img_float = np.float32(image) / 255
    
    # Image transformation
    transform = transforms.ToTensor()
    tensor = transform(img_float).unsqueeze(0)

    # Initialize list to store result images with captions
    result_images = []

    # Process each selected YOLO model
    for yolo_version in yolo_versions:
        # Load the model based on YOLO version
        model = load_yolo_model(yolo_version)
        target_layers = [model.model.model.model[-2]]  # Assumes last layer is used for Grad-CAM

        # Run YOLO detection
        results = model([rgb_img])
        boxes, colors, names = parse_detections(results)
        detections_img = draw_detections(boxes, colors, names, rgb_img.copy())

        # Grad-CAM visualization
        cam = EigenCAM(model, target_layers)
        grayscale_cam = cam(tensor)[0, :, :]
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # Renormalize Grad-CAM inside bounding boxes
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        renormalized_cam_image = show_cam_on_image(img_float, renormalized_cam, use_rgb=True)

        # Concatenate images and prepare the caption
        final_image = np.hstack((rgb_img, cam_image, renormalized_cam_image))
        caption = f"Results using {yolo_version}"
        result_images.append((Image.fromarray(final_image), caption))

    return result_images

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.CheckboxGroup(
            choices=["yolov3", "yolov5", "yolov7", "yolov8", "yolov10"],
            value=["yolov5"],  # Set the default value (YOLOv5 checked by default)
            label="Select Model(s)",
        )
    ],
    outputs = gr.Gallery(label="Results", elem_id="gallery", rows=2, height=500),
    title="Visualising the key image features that drive decisions with our explainable AI tool.",
    description="XAI: Upload an image to visualize object detection of your models.."
)

if __name__ == "__main__":
    interface.launch()
