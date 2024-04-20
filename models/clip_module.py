
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
    Make predictions on the images and texts

    Args:
    model: CLIP model
    processor: CLIP processor
    images: List of PIL images
    texts: List of texts
    device: Device to run the model on

    Returns:
    probs: Predicted probabilities
    preds: Predicted labels
    """

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    with autocast():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    probs = probs.cpu().detach().numpy()
    index = np.argmax(probs, axis=1)
    preds = [texts[i] for i in index]
    return probs, preds

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
