
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from torch.cuda.amp import autocast
import numpy as np
from torchvision import transforms

def load_images(image_paths):
    # Load images as PIL images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    return images


def clip_prediction(model, processor, images, texts, device):
    """
    Make predictions on the images and texts and return the top 3 probabilities with their corresponding images and labels.

    Args:
    model: CLIP model
    processor: CLIP processor
    images: List of PIL images
    texts: List of texts
    device: Device to run the model on

    Returns:
    top_probs: List of top 3 probabilities
    top_segments: List of images corresponding to the top 3 probabilities
    top_labels: List of labels corresponding to the top 3 probabilities
    """

    # Process the inputs
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device

    # Make predictions
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(**inputs)

    # Calculate probabilities
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we apply softmax to get probabilities
    probs = probs.cpu().detach().numpy()  # move data back to cpu and convert to numpy

    # Flatten the probability matrix to sort and find top 3 probabilities
    flat_indices = np.argsort(-probs.ravel())[:3]  # Get indices of top 3 probabilities in flattened array
    top_probs_indices = np.unravel_index(flat_indices, probs.shape)  # Convert flat indices to tuple (row, col)

    image_index = top_probs_indices[0][0]
    image_pred_index = top_probs_indices[1][0]
    image_pred = texts[image_pred_index]
    # Extract top 3 probabilities, segments, and labels
    top_probs = probs[top_probs_indices]
    top_segments = [images[idx] for idx in top_probs_indices[0]]
    top_labels = [texts[idx] for idx in top_probs_indices[1]]

    return image_index, image_pred





def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image_paths = ["/home/jr151/code/projects/Auto-Segment-Collage/sam/results/1b0b0b614b.jpg/segmented_image_0.png", "/home/jr151/code/projects/Auto-Segment-Collage/sam/results/1b0b0b614b.jpg/segmented_image_1.png"]  # Add your image paths here
    images = load_images(image_paths)
    
    texts = ["cat", "aantelope", "dog", "Background"] # Add your text here
    probs, preds = clip_prediction(model, processor, images, texts, device)
    print(probs)
    print(preds)

if __name__ == '__main__':
    main()
