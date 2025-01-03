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

# CSS to style the Gradio components and HTML content
custom_css = """
body {
    background-position: center;
    background-size: cover;
    background-attachment: fixed;
    height: 100%;  /* Ensure body height is 100% of the viewport */
    margin: 0;
    overflow-y: auto;  /* Allow vertical scrolling */
}
.custom-row {
    display: flex;
    justify-content: center;
    padding: 20px;
}
.custom-button {
    background-color: #6a1b9a;
    color: white;
    font-size: 14px;  /* Reduced font size */
    width: 120px;     /* Reduced width */
    border-radius: 8px;
    cursor: pointer;
}
/* Custom border styles for all Gradio components */
.gradio-container, .gradio-row, .gradio-column, .gradio-input, .gradio-image, .gradio-checkgroup, .gradio-button, .gradio-markdown {
    border: 3px solid black !important;  /* Border width and color */
    border-radius: 8px !important;      /* Rounded corners */
}
/* Additional customizations for images to enhance visibility of the border */
.gradio-image img {
    border-radius: 8px !important;
    border: 3px solid black !important;  /* Border for image */
}
/* Custom Row for images and buttons */
.custom-row img {
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}
#highlighted-text {
    font-weight: bold;
    color: #1976d2;
}
.gradio-block {
    max-height: 100vh;  /* Allow scrolling within the Gradio blocks */
    overflow-y: auto;   /* Enable scrolling for the content if it overflows */
}
#neural-vista-title {
    color: purple !important;  /* Purple color for the title */
    font-size: 32px;           /* Adjust font size as needed */
    font-weight: bold;
    text-align: center;
}
#neural-vista-text {
    color: purple !important;  /* Purple color for the title */
    font-size: 14px;           /* Adjust font size as needed */
    font-weight: bold;
    text-align: center;
}
"""

# Then in the Gradio interface:

with gr.Blocks(css=custom_css) as interface:
   
    gr.HTML("""
    <span style="color: #E6E6FA; font-weight: bold;" id="neural-vista-title">NeuralVista</span><br>
    
    A powerful tool designed to help you <span style="color: #E6E6FA; font-weight: bold;" id="neural-vista-text">visualize</span> models in action.
""")
    
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
            #with gr.Row(elem_classes="custom-row"):
            run_button = gr.Button("Run", elem_classes="custom-button")


        with gr.Column():
            sample_display = gr.Image(
                value=load_sample_image(default_sample),  
                label="Selected Sample Image",
            )


    gr.HTML("""The visualization demonstrates object detection and interpretability. Detected objects are highlighted with bounding boxes, while the heatmap reveals regions of focus, offering insights into the model's decision-making process.</span>""")
    # Results and visualization
    with gr.Row(elem_classes="custom-row"):
        result_gallery = gr.Gallery(
            label="Results",
            rows=1, 
            height="auto",       # Adjust height automatically based on content
            columns=1 ,
            object_fit="contain"
        ) 
        netron_display = gr.HTML(label="Netron Visualization")

    # Update sample image
    sample_selection.change(
        fn=load_sample_image,
        inputs=sample_selection,
        outputs=sample_display,
    )
    
    gr.HTML("""
    <span style="color: purple; font-weight: bold;">Concept Discovery</span> involves identifying interpretable high-level features or concepts within a deep learning model's representation. It aims to understand what a model has learned and how these learned features relate to meaningful attributes in the data.<br><br>
    <span style="color: purple; font-weight: bold;">Deep Feature Factorization (DFF)</span> is a technique that decomposes the deep features learned by a model into disentangled and interpretable components. It typically involves matrix factorization methods applied to activation maps, enabling the identification of semantically meaningful concepts captured by the model.
    Together, these methods enhance model interpretability and provide insights into the decision-making process of neural networks.
""")
    with gr.Row(elem_classes="custom-row"):
        dff_gallery = gr.Gallery(
            label="Deep Feature Factorization",
            rows=2,          # 8 rows
            columns=4,       # 1 image per row
            object_fit="fit",
            height="auto"    # Adjust as needed
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
            gr.HTML("""
            Generated abstract visualizations of model""")
            netron_html = view_model(selected_models)

        # Launch threads
        t1 = threading.Thread(target=process_thread)
        t2 = threading.Thread(target=netron_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        image1, text, image2 = results[0]
        if isinstance(image2, list):
            # Check if image2 contains exactly 8 images
            if len(image2) == 8:
                print("image2 contains 8 images.")
            else:
                print("Warning: image2 does not contain exactly 8 images.")
        else:
            print("Error: image2 is not a list of images.")
        return [(image1, text)], netron_html, image2

    # Run button click
    run_button.click(
        fn=run_both,
        inputs=[sample_selection, upload_image, selected_models],
        outputs=[result_gallery, netron_display, dff_gallery],
    )

# Launch Gradio interface
if __name__ == "__main__":
    interface.launch(share=True)
