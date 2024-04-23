import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np

def segment_image(IMAGE_PATH, mask_generator, result_dir):
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)     # image_rgb is a numpy array with shape [W,H,3]
    sam_result = mask_generator.generate(image_rgb)

    masks = [
    mask['segmentation']
    for mask
    in sorted(sam_result, key=lambda x: x['area'], reverse=True)] 
    save_dir = result_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    segmented_images = []
    save_paths = []

    for i, mask in enumerate(masks[:10]):
        mask_3d = np.stack([mask]*3, axis=-1)
        # Use white background instead of black
        white_background = np.ones_like(image_rgb) * 255  # Create a white background image
        segmented_image = np.where(mask_3d, image_rgb, white_background)
        segmented_images.append(segmented_image)
        save_path = os.path.join(save_dir, f'segmented_image_{i}.png')
        save_paths.append(save_path)
        cv2.imwrite(save_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    
    return segmented_images, save_paths


