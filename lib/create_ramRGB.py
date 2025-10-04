import os
import numpy as np
import PIL.Image

out_put_dir = "./data/random_rgb_dataset"
num_images = 10000
image_size = (3, 224, 224)  # RGB images of size

# Create output directory if it doesn't exist
os.makedirs(out_put_dir, exist_ok=True)

for i in range(num_images):
    image = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
    # Transpose to HWC format (height, width, channel) before converting to PIL Image
    image = np.transpose(image, (1, 2, 0))
    image = PIL.Image.fromarray(image)
    image.save(os.path.join(out_put_dir, f"image_{i:05d}.jpg"), "JPEG", quality=95)