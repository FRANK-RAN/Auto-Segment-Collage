import os
import torch
from transformers import CLIPProcessor, CLIPModel
from models.clip_module import clip_prediction, load_images
from models.sam_module import segment_image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import shutil

def run_pipeline(image_paths, mask_generator, clip_model, clip_processor, device, label_texts, segment_result_dir, selected_images_dir):
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
        base_name = os.path.basename(image_path).split('.')[0]  # Use the base name of the original image file without extension
        result_dir = os.path.join(segment_result_dir, base_name)            # create a directory for each image
        os.makedirs(result_dir, exist_ok=True)
        # Segment the image
        segmented_images, save_paths = segment_image(image_path, mask_generator, result_dir)

        # Load and preprocess segmented images for CLIP
        processed_images = load_images(save_paths)

        # Use CLIP to label the images
        image_index, image_pred = clip_prediction(clip_model, clip_processor, processed_images, label_texts, device)
        selected_image_path = save_paths[image_index]
        if image_pred != 'background' or "unknown":
            if image_pred not in animal_images:
                animal_images[image_pred] = []
            unique_file_name = f"{base_name}_{os.path.basename(selected_image_path)}"
            # Copy the file
            selected_image_path_in_collection = os.path.join(selected_images_dir, unique_file_name)
            shutil.copy(selected_image_path, selected_image_path_in_collection)
            animal_images[image_pred].append(selected_image_path_in_collection)
        


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
    CLIP_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(CLIP_device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    label_texts = ["bear", "dog", "lion","panda", "elephant", "horse", "owl","tiger", "wolf","antelope", "background", "unknown"]

   
    image_paths = get_file_paths(dir_path)

    SAM_CHECKPOINT_PATH = 'sam_model/weights/sam_vit_h_4b8939.pth'      # TODO: Replace with the path to the checkpoint file
    SAM_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=SAM_device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    segment_result_dir = 'results/segmented_images'
    selected_images_dir = 'results/selected_images'
    os.makedirs(selected_images_dir, exist_ok=True)
    os.makedirs(segment_result_dir, exist_ok=True)
    clear_directory(segment_result_dir)
    clear_directory(selected_images_dir)

    animal_images = run_pipeline(image_paths, mask_generator, clip_model, clip_processor, CLIP_device, label_texts, segment_result_dir, selected_images_dir)
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

