import gradio as gr
import netron
import os
import threading
import time
from PIL import Image
import cv2
import numpy as np
import torch
from yolov5 import xai_yolov5
from yolov8 import xai_yolov8s

# Sample images directory
sample_images = {
    "Sample 1": os.path.join(os.getcwd(), "data/xai/sample1.jpeg"),
    "Sample 2": os.path.join(os.getcwd(), "data/xai/sample2.jpg"),
}

# Preloaded model file path (update this path as needed)
preloaded_model_file = os.path.join(os.getcwd(), "weight_files/yolov5.onnx")  # Example path

def load_sample_image(sample_name):
    """Load a sample image based on user selection."""
    image_path = sample_images.get(sample_name)
    if image_path and os.path.exists(image_path):
        return Image.open(image_path)
    return None

def process_image(sample_choice, uploaded_image, yolo_versions):
    """Process the image using selected YOLO models."""
    if uploaded_image is not None:
        image = uploaded_image  # Use the uploaded image
    else:
        image = load_sample_image(sample_choice)  # Use selected sample image

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

def serve_netron(model_file):
    """Start the Netron server in a separate thread."""
    threading.Thread(target=netron.start, args=(model_file,), daemon=True).start()
    time.sleep(1)  # Give some time for the server to start
    return "http://localhost:8080"  # Default Netron URL
def view_model():
    """Handle model visualization using preloaded model file."""
    if not os.path.exists(preloaded_model_file):
        return "Model file not found."
    
    netron_url = serve_netron(preloaded_model_file)
    return f'<iframe src="{netron_url}" width="100%" height="600px"></iframe>'

# Custom CSS for styling (optional)
custom_css = """
#run_button {
    background-color: purple;
    color: white;
    width: 120px;
    border-radius: 5px;
    font-size: 14px;
}
"""

with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("# XAI: Visualize Object Detection of Your Models")
    
    default_sample = "Sample 1"

    with gr.Row():
        # Left side: Sample selection and upload image
        with gr.Column():
            sample_selection = gr.Radio(
                choices=list(sample_images.keys()),
                label="Select a Sample Image",
                type="value",
                value=default_sample,
            )

            upload_image = gr.Image(
                label="Upload an Image",
                type="pil",  
            )

            selected_models = gr.CheckboxGroup(
                choices=["yolov5", "yolov8s"],
                value=["yolov5"],
                label="Select Model(s)",
            )

            run_button = gr.Button("Run", elem_id="run_button")

        with gr.Column():
            sample_display = gr.Image(
                value=load_sample_image(default_sample),  
                label="Selected Sample Image",
            )

    # Below the sample image, display results and architecture side by side
    with gr.Row():
        result_gallery = gr.Gallery(
            label="Results",
            elem_id="gallery",
            rows=1,
            height=500,
        )

        netron_display = gr.HTML(label="Netron Visualization")

    sample_selection.change(
        fn=load_sample_image,
        inputs=sample_selection,
        outputs=sample_display,
    )

    run_button.click(
        fn=process_image,
        inputs=[sample_selection, upload_image, selected_models],
        outputs=[result_gallery],
    )

    # Update Netron display when the interface loads
    netron_display.value = view_model()  # Directly set the value

# Launching Gradio app and handling Netron visualization separately.
if __name__ == "__main__":
    interface.launch(share=True)
