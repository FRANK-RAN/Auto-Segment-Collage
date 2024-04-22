import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
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

    for i, mask in enumerate(masks[:20]):
        mask_3d = np.stack([mask]*3, axis=-1)
        segmented_image = np.where(mask_3d, image_rgb, 0)  # Replace '0' with another value for a different background color
        segmented_images.append(segmented_image)
        save_path = os.path.join(save_dir, f'segmented_image_{i}.png')
        save_paths.append(save_path)
        cv2.imwrite(save_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    
    return segmented_images, save_paths

def segment_batch_images(IMAGE_DIR, mask_generator, result_dir='/home/jr151/code/projects/Auto-Segment-Collage/sam/results/'):
    for pic in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, pic)
        segment_image(image_path, mask_generator, result_dir)


def main():
    # HOME = os.getcwd()
    # Model_dir = os.path.join('/home/jr151/model')
    # CHECKPOINT_PATH = os.path.join(Model_dir, "seg", "sam_vit_h_4b8939.pth")
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_PATH = '/home/jr151/model/seg/sam_vit_h_4b8939.pth'      # TODO: Replace with the path to the checkpoint file
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    segment_image('/home/jr151/code/projects/Auto-Segment-Collage/sam/data/dog-2.jpeg', mask_generator)
    segment_batch_images('/home/jr151/data/animals/animals/antelope', mask_generator)

if __name__ == '__main__':
    main()