import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import gradio as gr

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


def xai_yolov5(image):
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

    # Combine results
    final_image = np.hstack((image, detections_img, renormalized_cam_image))
    caption = "Results using YOLOv5"
    return Image.fromarray(final_image), caption


    
