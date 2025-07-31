# model/model_loader.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ChestXRayModel:
    def __init__(self):
        # ✅ Load ResNet-50 pretrained on ImageNet
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features

        # ✅ Simulate CheXpert's 14 class output
        self.model.fc = nn.Linear(num_features, 14)
        self.model.eval()

        # ✅ Simulated CheXpert condition labels
        self.classes = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]

        # ✅ Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image):
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.sigmoid(output).numpy()[0]

        threshold = 0.5
        predictions = [
            (cls, float(prob))
            for cls, prob in zip(self.classes, probs)
            if prob > threshold
        ]

        return predictions
