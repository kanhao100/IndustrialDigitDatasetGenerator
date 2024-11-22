import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Union
from font_png_augmentation import ImageAugmentor
from default_config import DEFAULT_CONFIG

def copy_font_directory(src_dir: str, dest_dir: str):
    """复制字体目录到目标位置
    
    Args:
        src_dir: 源字体目录
        dest_dir: 目标目录
    """
    # 如果目标目录已存在，先删除
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    
    # 复制目录
    shutil.copytree(src_dir, dest_dir)
    print(f"已复制字体目录从 {src_dir} 到 {dest_dir}")

def generate_single_digit_image(
    digit_path: str,
    augmentor: ImageAugmentor,
    output_size: int = 256
) -> np.ndarray:
    """生成包含单个数字的增强图像
    
    Args:
        digit_path: 数字图像路径
        augmentor: 图像增强器实例
        output_size: 输出图像大小
        
    Returns:
        np.ndarray: 增强后的图像数组
    """
    # 加载数字图像
    digit_img = augmentor._load_digit(digit_path)
    
    # 调整大小
    digit_size = int(output_size * random.uniform(0.4, 0.6))  # 适当调大数字尺寸
    digit_img = augmentor._resize_digit(digit_img, digit_size)
    
    # 应用增强
    if random.random() < augmentor.augmentation_prob:
        digit_img = augmentor._apply_augmentations(digit_img)
    
    return digit_img
    # # 创建画布
    # canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
    
    # # 计算居中位置
    # x = (output_size - digit_size) // 2
    # y = (output_size - digit_size) // 2
    
    # # 将数字放置在画布中心
    # mask = digit_img < 255
    # canvas[y:y+digit_size, x:x+digit_size][mask] = digit_img[mask]
    
    # 添加工业背景
    # canvas = augmentor._add_industrial_background(canvas)
    
    # return canvas

def generate_classification_dataset(
    src_dir: str,
    output_dir: str,
    images_per_class: int,
    augmentor: ImageAugmentor,
    seed: int = None
):
    """生成用于分类的数据集
    
    Args:
        src_dir: 源字体目录
        output_dir: 输出目录
        images_per_class: 每个类别生成的图像数量
        augmentor: 图像增强器实例
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个数字创建子目录
    for digit in range(10):
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
    
    # 获取每个数字的所有字体图像
    digit_files = {}
    for digit in range(10):
        digit_path = os.path.join(src_dir, str(digit))
        if os.path.exists(digit_path):
            digit_files[str(digit)] = [
                os.path.join(digit_path, f) 
                for f in os.listdir(digit_path) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
    
    # 生成数据集
    print("开始生成分类数据集...")
    for digit, files in digit_files.items():
        print(f"\n处理数字 {digit}")
        output_digit_dir = os.path.join(output_dir, digit)
        
        for i in tqdm(range(images_per_class), desc=f"数字 {digit}"):
            # 随机选择一个字体文件
            digit_file = random.choice(files)
            
            # 生成增强图像
            image = generate_single_digit_image(digit_file, augmentor)
            
            # 保存图像
            image_path = os.path.join(output_digit_dir, f"{digit}_{i:06d}.png")
            Image.fromarray(image).save(image_path)

def main():
    # 源目录和目标目录
    src_dir = "font_numbers"
    work_dir = "classification_dataset_work"
    output_dir = "classification_dataset"
    
    # 复制字体目录
    # copy_font_directory(src_dir, work_dir)
    
    # 创建增强器实例
    config = DEFAULT_CONFIG.copy()
    augmentor = ImageAugmentor(**config)
    
    # 生成数据集
    generate_classification_dataset(
        src_dir=src_dir,
        output_dir=output_dir,
        images_per_class=1500,  # 每个数字生成1000张图像
        augmentor=augmentor,
        seed=42
    )
    
    # 清理工作目录
    # shutil.rmtree(work_dir)
    # print("\n数据集生成完成！")

if __name__ == "__main__":
    main() 