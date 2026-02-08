import os
from PIL import Image
import torch
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path):
    return Image.open(path).convert("RGB")

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_resnet_model():
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # feature extractor
    model.eval()
    model.to(DEVICE)
    return model
