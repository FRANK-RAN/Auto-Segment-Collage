import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel
from models.clip_module import clip_prediction, load_images
from models.sam_module import segment_image
import os
import torch
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from torchvision import transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import shutil

def run_pipeline(image_paths, mask_generator, clip_model, clip_processor, device, label_texts, segment_result_dir):
    """
    Pipeline to process a list of image paths: segment, label, and filter images based on labels.

    Args:
    image_paths: List of paths to images.
    mask_generator: SAM model for generating masks.
    clip_model: CLIP model for image labeling.
    clip_processor: CLIP processor for preprocessing inputs for CLIP.
    device: Computation device ('cuda' or 'cpu').
    label_texts: List of labels for classification (80 animals + 'background').

    Returns:
    A dictionary with animal labels as keys and lists of segmented image paths as values.
    """
    animal_images = {}

    # Process each image
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        result_dir = os.path.join(segment_result_dir, base_name)            # create a directory for each image
        os.makedirs(result_dir, exist_ok=True)

        # Segment the image
        segmented_images, save_paths = segment_image(image_path, mask_generator, result_dir)

        # Load and preprocess segmented images for CLIP
        # processed_images = [transform(Image.fromarray(img.astype(np.uint8)).convert("RGB")) for img in segmented_images]
        processed_images = load_images(save_paths)

        # Use CLIP to label the images
        probs, preds = clip_prediction(clip_model, clip_processor, processed_images, label_texts, device)

        # Filter and organize results
        for pred, path in zip(preds, save_paths):
            if pred != 'Background':  # Assuming label_texts include specific animal names
                if pred not in animal_images:
                    animal_images[pred] = []
                animal_images[pred].append(path)

    return animal_images

def get_file_paths(directory):
    """
    Collect all file paths in the given directory and its subdirectories.

    Args:
    directory (str): The path to the directory where files are to be searched.

    Returns:
    list: A list containing all file paths found within the directory and its subdirectories.
    """
    file_paths = []  # List to store all file paths
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def process(dir_path):
    CLIP_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(CLIP_device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    label_texts = ["cat", "antelope", "bear", "background"]  # Add your text here
    image_paths = get_file_paths(dir_path)

    SAM_CHECKPOINT_PATH = 'checkpoints/sam_vit_h_4b8939.pth'      # TODO: Replace with the path to the checkpoint file
    SAM_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=SAM_device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    segment_result_dir = 'results'
    
    clear_directory(segment_result_dir)

    animal_images = run_pipeline(image_paths, mask_generator, clip_model, clip_processor, CLIP_device, label_texts, segment_result_dir)
    return animal_images


def get_file_paths(directory):
    """
    Collect all file paths in the given directory and its subdirectories.

    Args:
    directory (str): The path to the directory where files are to be searched.

    Returns:
    list: A list containing all file paths found within the directory and its subdirectories.
    """
    file_paths = []  # List to store all file paths
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


if __name__ == '__main__':
    animal_images = process("data")
    print(animal_images)