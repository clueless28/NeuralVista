import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
from yolov5 import xai_yolov5
from yolov8 import xai_yolov8s

sample_images = {
    "Sample 1": os.path.join(os.getcwd(), "data/xai/sample1.jpeg"),
    "Sample 2":  os.path.join(os.getcwd(), "data/xai/sample2.jpg"),
}
def load_sample_image(sample_name):
    image_path = sample_images.get(sample_name)
    if image_path and os.path.exists(image_path):
        return Image.open(image_path)
    return None

default_sample_image = load_sample_image("Sample 1")

def load_sample_image(choice):
    if choice in sample_images:
        image_path = sample_images[choice]
        return cv2.imread(image_path)[:, :, ::-1]  
    else:
        raise ValueError("Invalid sample selection.")


def process_image(sample_choice, uploaded_image, yolo_versions=["yolov5"]):
    print(sample_choice, upload_image)
    if uploaded_image is not None:
        image = uploaded_image  # Use the uploaded image
    else:
        # Otherwise, use the selected sample image
        image = load_sample_image(sample_choice)
    image = np.array(image)
    image = cv2.resize(image, (640, 640))
    result_images = []
    for yolo_version in yolo_versions:
        if yolo_version == "yolov5":
            result_images.append(xai_yolov5(image)) 
        elif yolo_version == "yolov8s":
            result_images.append(xai_yolov8s(image))
        else:
            result_images.append((Image.fromarray(image), f"{yolo_version} not yet implemented."))
    return result_images

with gr.Blocks() as interface:
    gr.Markdown("# XAI: Visualize Object Detection of Your Models")
    gr.Markdown("Select a sample image to visualize object detection.")
    default_sample = "Sample 1"
    with gr.Row():
        # Left side: Sample selection and upload image
        with gr.Column():
            sample_selection = gr.Radio(
                choices=list(sample_images.keys()),
                label="Select a Sample Image",
                type="value",
                value=default_sample,  # Set default selection
            )
            # Upload image below sample selection
            gr.Markdown("**Or upload your own image:**")
            upload_image = gr.Image(
                label="Upload an Image",
                type="filepath",  # Correct type for file path compatibility
            )
        # Right side: Selected sample image display
        sample_display = gr.Image(
            value=load_sample_image(default_sample),  
            label="Selected Sample Image",
        )
    
    sample_selection.change(
        fn=load_sample_image,
        inputs=sample_selection,
        outputs=sample_display,
    )

    selected_models = gr.CheckboxGroup(
        choices=["yolov5", "yolov8s"],
        value=["yolov5"],
        label="Select Model(s)",
    )
    result_gallery = gr.Gallery(label="Results", elem_id="gallery", rows=2, height=500)

    gr.Button("Run").click(
        fn=process_image,
        inputs=[sample_selection, upload_image, selected_models],  # Include both options
        outputs=result_gallery,
    )

interface.launch(share=True)