# model/gradcam.py

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook to get gradients from the final conv layer
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Hook to get activations from the final conv layer
        def forward_hook(module, input, output):
            self.activations = output

        # Register hooks on final conv layer
        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def generate_heatmap(self, image: Image.Image, class_idx: int = None):
        # Preprocess image
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        # Forward pass
        output = self.model(input_tensor)

        # If no class index given, use the one with highest score
        if class_idx is None:
            class_idx = output.squeeze().sigmoid().argmax().item()

        # Zero gradients and backprop for that class
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients[0].detach().numpy()     # [C, H, W]
        activations = self.activations[0].detach().numpy() # [C, H, W]

        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam) + 1e-8  # Avoid division by zero

        # Convert original image to numpy
        img_np = np.array(image.resize((224, 224)))
        if len(img_np.shape) == 2 or img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        # Create heatmap and overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Convert back to PIL
        final_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        return final_img, class_idx
