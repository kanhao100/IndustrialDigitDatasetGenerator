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

def add_stripe_interference_position(output_size: int) -> (int, int, int, int):
        """添加竖条条纹干扰位置
        
        Args:
            output_size: 输出图像大小
            
        Returns:
            rect_y: 矩形的y坐标
            rect_x: 矩形的x坐标
            rect_height: 矩形高度
            rect_width: 矩形宽度
        """
        rect_height = random.randint(10, 60)  # 矩形高度
        rect_width = random.randint(5, 20)  # 矩形宽度
        rect_color = random.randint(0, 30)  # 使用深色/黑色系，0-50的灰度值
        if random.random() < 0.5:
            rect_y = 0
            rect_x = random.randint(0, output_size - rect_width)
        else:
            rect_y = output_size - rect_height
            rect_x = random.randint(0, output_size - rect_width)
        return rect_y, rect_x, rect_height, rect_width, rect_color


def generate_single_digit_image(
    digit_path: str,
    augmentor: ImageAugmentor,
    output_size: int = 224
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
    

    if random.random() < 0.7:
        digit_img = augmentor._add_random_cropped_background(digit_img)
        return digit_img
    
    # 创建画布
    canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
    # 计算居中位置
    x = (output_size - digit_size) // 2
    y = (output_size - digit_size) // 2
    # 将数字放置在画布中心
    # mask = digit_img < 255
    canvas[y:y+digit_size, x:x+digit_size] = digit_img
    # 添加工业背景
    canvas = augmentor._add_random_cropped_background(canvas)
    if random.random() < 0.7:
        return canvas

    rect_y, rect_x, rect_height, rect_width, rect_color = add_stripe_interference_position(output_size)
    canvas[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width] = rect_color
    if random.random() < 0.25:
        return canvas
       
    rect_y, rect_x, rect_height, rect_width, rect_color = add_stripe_interference_position(output_size)
    canvas[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width] = rect_color
    if random.random() < 0.5:
        return canvas
    
    rect_y, rect_x, rect_height, rect_width, rect_color = add_stripe_interference_position(output_size)
    canvas[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width] = rect_color
    if random.random() < 0.5:
        return canvas
    
    rect_y, rect_x, rect_height, rect_width, rect_color = add_stripe_interference_position(output_size)
    canvas[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width] = rect_color
    return canvas

def generate_classification_dataset(
    input_dirs: Union[str, List[str]],
    output_dir: str,
    images_per_class: int,
    augmentor: ImageAugmentor,
    seed: int = None,
    dir_weights: List[float] = None
):
    """生成用于分类的数据集
    
    Args:
        input_dirs: 源字体目录或目录列表
        output_dir: 输出目录
        images_per_class: 每个类别生成的图像数量
        augmentor: 图像增强器实例
        seed: 随机种子
        dir_weights: 每个输入目录的权重
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 确保input_dirs是列表格式
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    
    # 如果没有提供权重，则默认每个目录的权重相同
    if dir_weights is None:
        dir_weights = [1.0] * len(input_dirs)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个数字创建子目录
    for digit in range(10):
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
    
    # 获取每个数字的所有字体图像
    digit_files = {str(digit): [] for digit in range(10)}
    for input_dir in input_dirs:
        for digit in range(10):
            digit_path = os.path.join(input_dir, str(digit))
            if os.path.exists(digit_path):
                digit_files[str(digit)].extend([
                    os.path.join(digit_path, f) 
                    for f in os.listdir(digit_path) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))
                ])
    
    # 生成数据集
    print("开始生成分类数据集...")
    for digit, files in digit_files.items():
        print(f"\n处理数字 {digit}")
        output_digit_dir = os.path.join(output_dir, digit)
        
        for i in tqdm(range(images_per_class), desc=f"数字 {digit}"):
            # 确保文件列表不为空
            if not files:
                print(f"警告: 没有找到数字 {digit} 的字体文件")
                continue
            
            # 根据目录权重随机选择一个字体文件
            selected_dir = random.choices(input_dirs, weights=dir_weights, k=1)[0]
            digit_path = os.path.join(selected_dir, str(digit))
            digit_files_in_dir = [
                os.path.join(digit_path, f) 
                for f in os.listdir(digit_path) 
                if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            
            # 确保目录中有文件
            if not digit_files_in_dir:
                print(f"警告: 目录 {digit_path} 中没有找到字体文件")
                continue
            
            digit_file = random.choice(digit_files_in_dir)
            
            # 生成增强
            image = generate_single_digit_image(digit_file, augmentor)
            
            # 保存图像
            image_path = os.path.join(output_digit_dir, f"{digit}_{i:06d}.png")
            Image.fromarray(image).save(image_path)

def main():
    # 源目录和目标目录
    input_dirs = ["font_numbers", "template_num"]
    output_dir = "classification_dataset"
    
    # 创建增强器实例
    config = DEFAULT_CONFIG.copy()
    augmentor = ImageAugmentor(**config)
    
    # 生成数据集
    generate_classification_dataset(
        input_dirs=input_dirs,
        output_dir=output_dir,
        images_per_class=2000,  # 每个数字生成100张图像
        augmentor=augmentor,
        seed=42,
        dir_weights=[0.8, 0.2]  # 设置目录权重
    )
    
    # 清理工作目录
    # shutil.rmtree(work_dir)
    # print("\n数据集生成完成！")

if __name__ == "__main__":
    main()