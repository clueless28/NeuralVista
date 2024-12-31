import netron
import threading
import gradio as gr
import os
from PIL import Image
import cv2
import numpy as np
from yolov5 import xai_yolov5
from yolov8 import xai_yolov8s

# Sample images directory
sample_images = {
    "Sample 1": os.path.join(os.getcwd(), "data/xai/sample1.jpeg"),
    "Sample 2": os.path.join(os.getcwd(), "data/xai/sample2.jpg"),
}

def load_sample_image(sample_name):
    """Load a sample image based on user selection."""
    image_path = sample_images.get(sample_name)
    if image_path and os.path.exists(image_path):
        return Image.open(image_path)
    return None

def process_image(sample_choice, uploaded_image, yolo_versions, target_lyr = -5, n_components = 8):
    """Process the image using selected YOLO models."""
    # Load sample or uploaded image
    if uploaded_image is not None:
        image = uploaded_image
    else:
        image = load_sample_image(sample_choice)

    # Preprocess image
    image = np.array(image)
    image = cv2.resize(image, (640, 640))
    result_images = []

    # Apply selected models
    for yolo_version in yolo_versions:
        if yolo_version == "yolov5":
            result_images.append(xai_yolov5(image, target_lyr = -5, n_components = 8)) 
        elif yolo_version == "yolov8s":
            result_images.append(xai_yolov8s(image))
        else:
            result_images.append((Image.fromarray(image), f"{yolo_version} not implemented."))
    return result_images

def view_model(selected_models):
    """Generate Netron visualization for the selected models."""
    netron_html = ""
    for model in selected_models:
        if model == "yolov5":
            netron_html = f"""
            <iframe 
                src="https://netron.app/?url=https://huggingface.co/FFusion/FFusionXL-BASE/blob/main/vae_encoder/model.onnx" 
                width="100%" 
                height="800" 
                frameborder="0">
            </iframe>
            """
    return netron_html if netron_html else "<p>No valid models selected for visualization.</p>"

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
    gr.Markdown("# NeuralVista: Visualize Object Detection of Your Models")
    
    # Default sample
    default_sample = "Sample 1"

    with gr.Row():
        # Left side: Sample selection and image upload
        with gr.Column():
            sample_selection = gr.Radio(
                choices=list(sample_images.keys()),
                label="Select a Sample Image",
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

    # Results and visualization
    with gr.Row():
        result_gallery = gr.Gallery(
            label="Results",
            rows=1,
            height=500,
        )

        netron_display = gr.HTML(label="Netron Visualization")

    # Update sample image
    sample_selection.change(
        fn=load_sample_image,
        inputs=sample_selection,
        outputs=sample_display,
    )
    with gr.Row():
        dff_gallery = gr.Gallery(
            label="Deep Feature Factorization",
            rows=1,
            height=800,
        )


    # Multi-threaded processing
    def run_both(sample_choice, uploaded_image, selected_models):
        results = []
        netron_html = ""

        # Thread to process the image
        def process_thread():
            nonlocal results
            target_lyr = -5 
            n_components = 8
            results = process_image(sample_choice, uploaded_image, selected_models, target_lyr = -5, n_components = 8)

        # Thread to generate Netron visualization
        def netron_thread():
            nonlocal netron_html
            netron_html = view_model(selected_models)

        # Launch threads
        t1 = threading.Thread(target=process_thread)
        t2 = threading.Thread(target=netron_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        image1, text, image2 = results[0]
        print('results', results)
        print('image2', image2)
        return [(image1, text)], netron_html, [image2]

    # Run button click

    run_button.click(
        fn=run_both,
        inputs=[sample_selection, upload_image, selected_models],
        outputs=[result_gallery, netron_display, dff_gallery],
    )

# Launch Gradio interface
if __name__ == "__main__":
    interface.launch(share=True)
