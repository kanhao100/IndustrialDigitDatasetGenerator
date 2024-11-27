import os
import random
from PIL import Image

def get_random_images(image_dir, num_images):
    """从目录及其子目录中随机选择图像 | Randomly select images from a directory and its subdirectories"""
    all_images = []
    for root, _, files in os.walk(image_dir):
        all_images.extend([os.path.join(root, img) for img in files if img.endswith(('png', 'jpg', 'jpeg', 'bmp'))])
    return random.sample(all_images, num_images)

def compose_images(image_paths, layout):
    """合成图像 | Compose images based on the given layout"""
    images = [Image.open(img_path) for img_path in image_paths]
    widths, heights = zip(*(img.size for img in images))

    max_width = min(sum(widths), 1920)
    max_height = min(sum(heights), 1080)

    try:
        grid_size = tuple(map(int, layout.split('x')))
    except ValueError:
        raise ValueError("Unsupported layout")

    grid_width = min(max_width, grid_size[0] * max(widths))
    grid_height = min(max_height, grid_size[1] * max(heights))

    composed_image = Image.new('RGB', (grid_width, grid_height))

    for i, img in enumerate(images):
        row = i // grid_size[0]
        col = i % grid_size[0]
        x_offset = col * (grid_width // grid_size[0])
        y_offset = row * (grid_height // grid_size[1])
        composed_image.paste(img, (x_offset, y_offset))

    return composed_image

def main():
    # image_dir = './augmented_dataset'
    # image_dir = './font_numbers'
    # image_dir = './NEU-DET'
    image_dir = './classification_dataset'
    output_dir = './docs'  # 设置输出目录 | Set output directory
    os.makedirs(output_dir, exist_ok=True)

    layouts = ['2x2']
    num_images = 4  # 根据最大布局选择图像数量 | Choose number of images based on the largest layout

    for layout in layouts:
        image_paths = get_random_images(image_dir, num_images)
        composed_image = compose_images(image_paths, layout)
        composed_image.save(os.path.join(output_dir, f'composed_{layout}.jpg'))

if __name__ == "__main__":
    main()