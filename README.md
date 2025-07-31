#🩺 MedGPT-X: AI Chest X-ray Diagnostic Companion

MedGPT-X is an AI-powered educational tool that accepts chest X-ray images and performs the following:

-  Predicts possible chest-related conditions using a ResNet-50-based model
-  Generates Grad-CAM heatmaps to visualize important regions of the X-ray
-  Provides medically inspired human-readable summaries for better understanding

> ⚠ **This tool is built for educational and demonstration purposes only. It does NOT provide real medical diagnosis.**

---

## Features

-  Uses a pre-trained `microsoft/resnet-50` model
-  Provides multi-label predictions (e.g., Cardiomegaly, Lung Opacity, etc.)
-  Applies Grad-CAM to highlight key activation areas in the image
-  Outputs handcrafted medical-style summaries with insights, suggestions, and next steps

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sura-ak/medgpt-x.git
cd medgpt-x
