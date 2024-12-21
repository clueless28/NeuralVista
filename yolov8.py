import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import gradio as gr
from ultralytics import YOLO


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
    final_image = np.hstack((image, detections_img, renormalized_cam_image))
    caption = "Results using YOLOv8"
    return Image.fromarray(final_image), caption