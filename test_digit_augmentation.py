import matplotlib.pyplot as plt
import os
import random

from font_png_augmentation import ImageAugmentor
from default_config import DEFAULT_CONFIG

def test_digit_augmentation():
    # 配置参数
    config = DEFAULT_CONFIG.copy()

    augmentor = ImageAugmentor(**config)
    
    # 为每个数字生成多个增强样本
    digits = range(10)  # 0-9
    samples_per_digit = 10
    rows = len(digits)
    cols = samples_per_digit
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 18))
    fig.suptitle('Digital enhanced sample', fontsize=16)
    
    input_dir = "font_numbers"  # 字体图像目录
    
    for row, digit in enumerate(digits):
        digit_dir = os.path.join(input_dir, str(digit))
        digit_files = os.listdir(digit_dir)
        
        for col in range(cols):
            # 随机选择一个字体文件
            digit_file = random.choice(digit_files)
            digit_path = os.path.join(digit_dir, digit_file)
            
            # 使用新的两步加载和调整大小的逻辑
            digit_img = augmentor._load_digit(digit_path)
            digit_size = 64  # 固定大小以便比较
            digit_img = augmentor._resize_digit(digit_img, digit_size)
            
            # 应用数据增强
            digit_img = augmentor._apply_augmentations(digit_img)
            
            # 显示图像
            ax = axes[row, col]
            ax.imshow(digit_img, cmap='gray')
            if col == 0:  # 只在每行第一个图案处显示数字
                ax.set_title(f'Digit {digit}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_single_digit_augmentations():
    """测试单个数字的不同增强效果"""
    # 配置参数
    config = DEFAULT_CONFIG.copy()

    augmentor = ImageAugmentor(**config)
    
    # 选择一个数字进行详细测试
    test_digit = 1 
    input_dir = "font_numbers"
    digit_dir = os.path.join(input_dir, str(test_digit))
    digit_files = os.listdir(digit_dir)
    digit_file = digit_files[3]
    digit_path = os.path.join(digit_dir, digit_file)
    
    # 显示4x4的增强效果
    rows, cols = 6, 6
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(f'Various enhanced effects for the digit {test_digit}', fontsize=16)
    
    digit_size = 64
    for i in range(rows):
        for j in range(cols):
            # 使用新的两步加载和调整大小的逻辑
            digit_img = augmentor._load_digit(digit_path)
            digit_img = augmentor._resize_digit(digit_img, digit_size)
            
            # 应用增强
            digit_img = augmentor._apply_augmentations(digit_img)
            
            ax = axes[i, j]
            ax.imshow(digit_img, cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试所有数字的增强效果
    # test_digit_augmentation()
    
    # 测试单个数字的多种增强效果
    test_single_digit_augmentations() 