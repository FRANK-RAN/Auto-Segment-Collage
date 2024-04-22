import os
import shutil

def create_custom_dataset(source_dir, target_dir, num_subdirs, num_images_per_subdir):
    """
    Create a custom dataset by copying a specified number of images from each of the first few subdirectories.

    Args:
    source_dir (str): Path to the source directory containing subdirectories for each category.
    target_dir (str): Path to the directory where the dataset will be stored.
    num_subdirs (int): Number of subdirectories to process.
    num_images_per_subdir (int): Number of images to copy from each subdirectory.
    """
    # Make sure the target directory exists, create if not
    os.makedirs(target_dir, exist_ok=True)

    # List all subdirectories in the source directory
    subdirs = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])[:num_subdirs]

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        selected_files = files[:num_images_per_subdir]

        # Create a directory for each animal category within the target directory
        target_subdir = os.path.join(target_dir, subdir)
        os.makedirs(target_subdir, exist_ok=True)

        # Copy selected files to the target subdirectory
        for file in selected_files:
            src_path = os.path.join(subdir_path, file)
            dst_path = os.path.join(target_subdir, file)
            shutil.copy(src_path, dst_path)

        print(f"Copied {len(selected_files)} images from {subdir} to {target_subdir}")

if __name__ == "__main__":
    source_path = "/home/jr151/data/animals/animals"  # Update the source path as required
    dataset_path = "/home/jr151/code/projects/Auto-Segment-Collage/input/custom_dataset_small"  # Path where the custom dataset will be stored
    create_custom_dataset(source_path, dataset_path, 5, 5)
