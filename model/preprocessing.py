import os

import torch
import torchvision.transforms as transforms
from PIL import Image

data_path = "images"
output_path = "image_tensors"

# Image preprocessing parameters
image_size = 224
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

# Preprocessing transformation
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ]
)


def process_image(image_path):
    try:
        img = Image.open(image_path)
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        return img_tensor
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return None


def save_tensor_to_file(tensor, output_file):
    try:
        torch.save(tensor, output_file)
        print(f"Image tensor saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving image tensor to '{output_file}': {e}")
