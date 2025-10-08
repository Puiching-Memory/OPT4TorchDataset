import os
import numpy as np
import PIL.Image
from tqdm import tqdm
from loguru import logger

def create_random_rgb_dataset(output_dir="./data/random_rgb_dataset", 
                               num_images=10000, 
                               image_size=(224, 224), 
                               num_classes=100):
    """
    Generate a random RGB dataset in ImageFolder format.
    
    Creates synthetic dataset of random RGB images organized in class subdirectories,
    compatible with torchvision.datasets.ImageFolder.
    
    Parameters:
        output_dir (str): Root directory for dataset (will contain class_XX subdirs)
        num_images (int): Total number of images to generate
        image_size (tuple): Image dimensions as (height, width)
        num_classes (int): Number of distinct classes
        
    Returns:
        str: Path to created dataset directory
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for class_id in range(num_classes):
        os.makedirs(os.path.join(output_dir, f"class_{class_id:02d}"), exist_ok=True)
    
    for i in tqdm(range(num_images), desc="Generating images"):
        image = np.random.randint(0, 256, size=(image_size[0], image_size[1], 3), dtype=np.uint8)
        image_pil = PIL.Image.fromarray(image, mode='RGB')
        
        class_id = i % num_classes
        image_filename = f"img_{i:05d}.jpg"
        image_path = os.path.join(output_dir, f"class_{class_id:02d}", image_filename)
        
        image_pil.save(image_path, "JPEG", quality=95)
    
    logger.info(f"Dataset created: {num_images} images, {num_classes} classes at {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    dataset_path = create_random_rgb_dataset(
        output_dir="./data/random_rgb_dataset",
        num_images=100 * 10,
        image_size=(512, 512),
        num_classes=100
    )
    
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(dataset_path)
    print(f"Dataset created with {len(dataset)} images")
    print(f"Classes: {dataset.classes}")
    print(f"Sample: {dataset[0]}")