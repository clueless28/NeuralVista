import gradio as gr
import os

# Paths for images
yolov5_result = os.path.join(os.getcwd(), "data/xai/yolov5.png")
yolov8_result = os.path.join(os.getcwd(), "data/xai/yolov8.png")
yolov5_dff = os.path.join(os.getcwd(), "data/xai/yolov5_dff.png")
yolov8_dff = os.path.join(os.getcwd(), "data/xai/yolov8_dff.png")

description_yolov5 = """
### Feature Focus Comparison
| Feature           | <span style="color: maroon;"><strong>Dogs</strong></span>                | <span style="color: maroon;"><strong>Cats</strong></span>                |
|-------------------|-----------------------------------|-------------------------------|
| **Face & Snout**  | Eyes, nose, and mouth for recognition | Sharp eyes, whiskers         |
| **Ears**          | Pointed or floppy shapes          | Pointed for identification   |
| **Body Shape**    | Legs, tail, and contour           | Compact, sitting posture     |
| **Fur Texture**   | Curly (poodles), smooth (corgis)  | N/A                           |
| **Tail & Paws**   | N/A                               | Often highlighted             |
### Common Errors
| Issue             | Description                       |
|-------------------|-----------------------------------|
| **Background**    | Irrelevant areas confused with key features |
| **Shared Features**| Overlapping fur or body shapes causing errors |
### Insights:
- Visualizations help identify key traits and potential classification biases.
"""

description_yolov8 = """
### Feature Focus Comparison

| Feature             | **Dogs**                              | **Cats**                          |
|---------------------|---------------------------------------|-----------------------------------|
| **Facial Features**  | Eyes, nose, mouth for species ID      | Sharp focus on eyes and whiskers  |
| **Ears & Fur Texture**| Fluffy/smooth fur, pointed/floppy ears | N/A                               |
| **Body & Legs**      | Focus on contour, legs, and tails     | Emphasizes compact size and tail  |
| **Paws & Posture**   | N/A                                   | Sitting posture, paw structures  |

### Common Errors

| Issue               | Description                           |
|---------------------|---------------------------------------|
| **Background Focus**| Attention to irrelevant background regions |
| **Shared Features**  | Overlapping features between dogs and cats |
| **Edge Effects**     | Bias from emphasis on image borders during training |

### Insights:
- Attention-based mechanisms can improve focus on key features and reduce misclassification.
"""



# Netron HTML templates
def get_netron_html(model_url):
    return f"""
        <div style="background-color: black; padding: 1px; border: 0.5px solid white;">
            <iframe 
                src="{model_url}" 
                width="100%" 
                height="800" 
                frameborder="0">
            </iframe>
        </div>
    """

# URLs for Netron visualizations
yolov5_url = "https://netron.app/?url=https://huggingface.co/FFusion/FFusionXL-BASE/blob/main/vae_encoder/model.onnx"
yolov8_url = "https://netron.app/?url=https://huggingface.co/spaces/BhumikaMak/NeuralVista/resolve/main/weight_files/yolov8s.pt"

custom_css = """
body {
    background-color: white; 
    background-size: 1800px 1800px;
    height: 100%;
    color-scheme: light !important;
    margin: 0;
    overflow-y: auto;
}
#neural-vista-title {
    color: #800000 !important;
    font-size: 32px;
    font-weight: bold;
    text-align: center;
}
#neural-vista-text {
    color: #800000  !important;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
#highlighted-text {
    font-weight: bold;
    color: #1976d2;
}
.custom-row {
    display: flex;
    justify-content: center; /* Align horizontally */
    align-items: center;     /* Align vertically */
    padding: 10px;           /* Adjust as needed for spacing */
}
.custom-button {
    background-color: #800000;
    color: white;
    font-size: 12px;         /* Small font size */
    width: 100px !important;            /* Fixed width */
    height: 35px !important;            /* Fixed height */
    border-radius: 6px;      /* Slightly rounded corners */
    padding: 0 !important;              /* Remove extra padding */
    cursor: pointer;
    text-align: center;
    margin: 0 auto;          /* Center within its container */
    box-sizing: border-box;  /* Ensure consistent sizing */
}
#run-button {
    background-color: #800000 !important;
    color: white !important;
    font-size: 12px !important;  /* Small font size */
    width: 100px !important;     /* Fixed width */
    height: 35px !important;     /* Fixed height */
    border-radius: 6px !important;
    padding: 0 !important;
    text-align: center !important;
    display: block !important;   /* Ensure block-level alignment */
    margin: 0 auto !important;   /* Center horizontally */
    box-sizing: border-box !important;
}
/* Custom border styles for all Gradio components */
.gradio-container, .gradio-row, .gradio-column, .gradio-input, .gradio-image, .gradio-checkgroup, .gradio-button, .gradio-markdown {
    border: 3px #800000 !important;  /* Border width and color */
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
    color: #800000 !important;  /* Purple color for the title */
    font-size: 32px;           /* Adjust font size as needed */
    font-weight: bold;
    text-align: center;
}
#neural-vista-text {
    color: #800000  !important;  /* Purple color for the title */
    font-size: 18px;           /* Adjust font size as needed */
    font-weight: bold;
    text-align: center;
    
}

"""

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
        if model=="yolov8s":
            netron_html = f"""
            <iframe 
                src="https://netron.app/?url=https://huggingface.co/spaces/BhumikaMak/NeuralVista/resolve/main/weight_files/yolov8s.pt" 
                width="100%" 
                height="800" 
                frameborder="0">
            </iframe>
            """
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

with gr.Blocks(css=custom_css, theme="default") as demo:
    gr.HTML("""
      <div style="border: 2px solid #a05252; padding: 20px; border-radius: 8px;">
        <span style="color: #800000; font-family: 'Papyrus', cursive; font-weight: bold; font-size: 32px;">NeuralVista</span><br><br>
        <span style="color: black; font-family: 'Papyrus', cursive; font-size: 18px;">A harmonious framework of tools <span style="color: red; font-family: 'Papyrus', cursive; font-size: 18px;">☼</span> designed to illuminate the inner workings of AI.</span>
      </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(""" ## Yolov5 """)
            html_content1 = """
                <div style="display: flex; gap: 10px;">
                  <a href="https://github.com/ultralytics/yolov5/actions" target="_blank">
                    <img src="https://img.shields.io/badge/YOLOv5%20CI-passing-brightgreen" alt="YOLOv5 CI">
                  </a>
                  <a href="https://doi.org/10.5281/zenodo.7347926" target="_blank">
                    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7347926-blue" alt="DOI">
                  </a>
                  <a href="https://hub.docker.com/r/ultralytics/yolov5" target="_blank">
                    <img src="https://img.shields.io/badge/docker%20pulls-361k-blue" alt="Docker Pulls">
                  </a>
                </div>
            """
            gr.HTML(html_content1)
           # gr.HTML(get_netron_html(yolov5_url))
            gr.Image(yolov5_result, label="Detections & Interpretability Map")
            gr.Markdown(description_yolov5)
            gr.Image(yolov5_dff, label="Feature Factorization & discovered concept")
            

        with gr.Column():
            gr.Markdown(""" ## Yolov8s """)
            html_content2 = """
                <div style="display: flex; gap: 10px;">
                  <a href="https://github.com/ultralytics/ultralytics/actions" target="_blank">
                    <img src="https://img.shields.io/badge/YOLOv8%20CI-passing-brightgreen" alt="YOLOv8 CI">
                  </a>
                  <a href="https://zenodo.org/records/10443804" target="_blank">
                    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7347926-blue" alt="DOI">
                  </a>
                  <a href="https://hub.docker.com/r/ultralytics/ultralytics" target="_blank">
                    <img src="https://img.shields.io/badge/docker%20pulls-500k-blue" alt="Docker Pulls">
                  </a>
                </div>
            """
            gr.HTML(html_content2)
           # gr.HTML(get_netron_html(yolov8_url))
            gr.Image(yolov8_result, label="Detections & Interpretability Map")
            gr.Markdown(description_yolov8)
            gr.Image(yolov8_dff, label="Feature Factorization & discovered concept")

    gr.HTML(
        """
        <div style="text-align: center; border: 3px solid maroon; padding: 10px; border-radius: 10px; background-color: #f8f8f8;">
            <h3>Want to try yourself? 🚀</h3>
            <p><b>Upload an image below to discover <span style="color: #ff6347;">☼</span> the concepts</b></p>
        </div>
        """
    )
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
                choices=["yolov5", "yolov8s"],  # Only the models that can be selected
                value=["yolov5"],
                label="Select Model(s)",
            )
            run_button = gr.Button("Run", elem_id="run-button")

        with gr.Column():
            sample_display = gr.Image(
                value=load_sample_image(default_sample),  
                label="Selected Sample Image",
            )
    
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
    gr.Markdown(""" #### Feature Factorization & discovered concepts. """)
    with gr.Row(elem_classes="custom-row"):
        dff_gallery = gr.Gallery(
            label="Feature Factorization & discovered concept",
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
            

demo.launch()
