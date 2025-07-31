# app.py

import gradio as gr
from PIL import Image

from model_loader import ChestXRayModel
from gradcam import GradCAM
from summarizer import MedicalSummarizer


# Initialize components
xray_model = ChestXRayModel()
gradcam = GradCAM(xray_model.model)  # pass internal resnet model
summarizer = MedicalSummarizer()

def analyze_xray(image: Image.Image):
    # Step 1: Predict conditions
    predictions = xray_model.predict(image)

    if not predictions:
        summary_text = "✅ No conditions detected. The chest X-ray appears normal."
        cam_image = image  # Just show the original image if nothing found
        prediction_text = "Model did not detect any abnormalities."
    else:
        # Step 2: Generate Grad-CAM for top prediction
        top_condition, _ = predictions[0]
        class_idx = xray_model.classes.index(top_condition)
        cam_image, _ = gradcam.generate_heatmap(image, class_idx=class_idx)

        # Step 3: Prepare readable prediction results
        prediction_text = "\n".join([f"🔹 {label}: {score * 100:.1f}%" for label, score in predictions])

        # Step 4: Generate AI medical summary
        summary_text = summarizer.summarize(predictions)

    return cam_image, prediction_text, summary_text

# Gradio UI setup
title = " 🩺 MedGPT-X: AI Chest X-ray Diagnostic Companion"
description = "Upload a chest X-ray image to get AI-powered condition predictions, heatmap visualization (Grad-CAM), and a human-readable medical summary."

iface = gr.Interface(
    fn=analyze_xray,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM Heatmap"),
        gr.Textbox(label="Predicted Conditions"),
        gr.Markdown(label="AI Medical Summary")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True)


