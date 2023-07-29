import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# Assuming you have already fine-tuned the model and it's saved at 'output_dir'.
output_dir = "./fine_tuned_clip_model"

# Replace these with your preprocessed image and text data
# Make sure they are stored as lists of tensors and strings, respectively
image_data = [...]  # List of image tensors
text_data = [...]   # List of text descriptions

# Function to preprocess a single image and convert it to a tensor
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def find_most_relevant_texts(image_path, clip_model, processor, text_data):
    # Preprocess the input image
    img_tensor = preprocess_image(image_path)

    # Encode the image tensor
    with torch.no_grad():
        img_features = clip_model.encode_image(img_tensor)

    # Encode the text data
    with torch.no_grad():
        text_features = clip_model.encode_text(text_data)

    # Calculate similarity scores between the image and text features
    similarity_scores = F.cosine_similarity(img_features, text_features)

    # Sort the text data based on similarity scores in descending order
    sorted_indices = torch.argsort(similarity_scores, descending=True)

    # Get the top 5 most relevant texts
    top_relevant_texts = [text_data[idx] for idx in sorted_indices[:5]]
    return top_relevant_texts

def find_most_relevant_images(text, clip_model, processor, image_data):
    # Encode the input text
    with torch.no_grad():
        text_feature = clip_model.encode_text(text)

    # Encode the image data
    with torch.no_grad():
        image_features = clip_model.encode_image(image_data)

    # Calculate similarity scores between the text and image features
    similarity_scores = F.cosine_similarity(text_feature, image_features)

    # Sort the image data based on similarity scores in descending order
    sorted_indices = torch.argsort(similarity_scores, descending=True)

    # Get the top 5 most relevant images
    top_relevant_images = [image_data[idx] for idx in sorted_indices[:5]]
    return top_relevant_images

# Load the fine-tuned CLIP model and processor
clip_model = CLIPModel.from_pretrained(output_dir)
processor = CLIPProcessor.from_pretrained(output_dir)

# Example usage to find most relevant texts given an image
example_image_path = "path/to/your/image.jpg"
most_relevant_texts = find_most_relevant_texts(example_image_path, clip_model, processor, text_data)
print("Most relevant texts for the given image:")
for idx, text in enumerate(most_relevant_texts, 1):
    print(f"{idx}. {text}")

# Example usage to find most relevant images given some text
example_text = "A beautiful sunset over the mountains."
most_relevant_images = find_most_relevant_images(example_text, clip_model, processor, image_data)

# Display the most relevant images
plt.figure(figsize=(15, 10))
plt.suptitle("Most Relevant Images for the Given Text")

for i, img_tensor in enumerate(most_relevant_images, 1):
    plt.subplot(1, 5, i)
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.axis("off")

plt.show()
