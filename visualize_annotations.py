import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import Tuple

def get_char_label(class_id: int) -> str:
    """根据类别ID获取对应的字符标签"""
    if 0 <= class_id <= 9:  # 数字 0-9
        return str(class_id)
    elif 10 <= class_id <= 35:  # 大写字母 A-Z
        return chr(ord('A') + (class_id - 10))
    elif 36 <= class_id <= 61:  # 小写字母 a-z
        return chr(ord('a') + (class_id - 36))
    else:
        return "Unknown"

def get_color(class_id: int, digit_colors: np.ndarray) -> Tuple[int, int, int]:
    """获取类别对应的颜色
    Args:
        class_id: 类别ID
        digit_colors: 数字0-9的专用颜色数组
    Returns:
        Tuple[int, int, int]: RGB颜色值
    """
    if 0 <= class_id <= 9:
        # 数字使用专用颜色
        return tuple(map(int, digit_colors[class_id]))
    else:
        # 其他字符复用数字的颜色
        return tuple(map(int, digit_colors[class_id % 10]))

def visualize_yolo_annotations(image_path: str, label_path: str, return_image: bool = False):
    """可视化YOLO格式的标注，支持数字和字母"""
    # 读取图像
    img = Image.open(image_path)
    width, height = img.size
    
    # 转换为RGB以便绘制彩色标注
    vis_img = img.convert('RGB')
    draw = ImageDraw.Draw(vis_img)
    
    # 读取标注文件
    with open(label_path, 'r') as f:
        annotations = f.readlines()
    
    # 为数字0-9生成不同的颜色
    digit_colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    digit_colors = (digit_colors[:, :3] * 255).astype(np.uint8)
    
    # 设置字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 绘制每个标注框
    for ann in annotations:
        # 解析YOLO格式的标注
        class_id, center_x, center_y, w, h = map(float, ann.strip().split())
        class_id = int(class_id)
        
        # 转换为像素坐标
        center_x = int(center_x * width)
        center_y = int(center_y * height)
        box_w = int(w * width)
        box_h = int(h * height)
        
        # 计算框的坐标
        x1 = int(center_x - box_w/2)
        y1 = int(center_y - box_h/2)
        x2 = int(center_x + box_w/2)
        y2 = int(center_y + box_h/2)
        
        # 获取当前类别的颜色
        color = get_color(class_id, digit_colors)
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 获取字符标签
        label = get_char_label(class_id)
        
        # 添加标签背景
        # 使用textbbox获取文本边界框
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        label_w = right - left
        label_h = bottom - top
        draw.rectangle([x1, y1-label_h-4, x1+label_w+4, y1], fill=color)
        
        # 添加类别标签
        draw.text((x1+2, y1-label_h-2), label, 
                 fill=(255, 255, 255),  # 白色文字
                 font=font)
    
    # 修改显示逻辑
    if not return_image:
        plt.figure(figsize=(12, 12))
        plt.imshow(vis_img)
        plt.axis('off')
        plt.title('YOLO Annotations Visualization (Numbers and Letters)')
        plt.show()
    return vis_img  # 返回处理后的图像

def visualize_multiple_images(base_pattern: str, start_idx: int, num_images: int = 5):
    """横向显示多张图片的标注结果
    Args:
        base_pattern: 图片路径模式，例如 "./augmented_dataset/image_{:06d}"
        start_idx: 起始图片索引
        num_images: 要显示的图片数量
    """
    images = []
    max_height = 0
    total_width = 0
    
    # 处理每张图片
    for i in range(num_images):
        idx = start_idx + i
        image_path = f"{base_pattern.format(idx)}.png"
        label_path = f"{base_pattern.format(idx)}.txt"
        
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue
            
        img = visualize_yolo_annotations(image_path, label_path, return_image=True)
        images.append(img)
        max_height = max(max_height, img.size[1])
        total_width += img.size[0]
    
    # 创建拼接图片
    result = Image.new('RGB', (total_width, max_height))
    current_x = 0
    
    # 拼接图片
    for img in images:
        result.paste(img, (current_x, 0))
        current_x += img.size[0]
    
    # 显示拼接结果
    plt.figure(figsize=(20, 4))
    plt.imshow(result)
    plt.axis('off')
    plt.title('Multiple Images Visualization')
    plt.show()

def visualize_multiple_raw_images(base_pattern: str, start_idx: int, num_images: int = 5):
    """横向显示多张原始图片（不包含标注）
    Args:
        base_pattern: 图片路径模式，例如 "./augmented_dataset/image_{:06d}"
        start_idx: 起始图片索引
        num_images: 要显示的图片数量
    """
    images = []
    max_height = 0
    total_width = 0
    
    # 处理每张图片
    for i in range(num_images):
        idx = start_idx + i
        image_path = f"{base_pattern.format(idx)}.png"
        
        if not os.path.exists(image_path):
            continue
            
        img = Image.open(image_path).convert('RGB')
        images.append(img)
        max_height = max(max_height, img.size[1])
        total_width += img.size[0]
    
    # 创建拼接图片
    result = Image.new('RGB', (total_width, max_height))
    current_x = 0
    
    # 拼接图片
    for img in images:
        result.paste(img, (current_x, 0))
        current_x += img.size[0]
    
    # 显示拼接结果
    plt.figure(figsize=(20, 4))
    plt.imshow(result)
    plt.axis('off')
    plt.title('Multiple Raw Images Visualization')
    plt.show()

if __name__ == "__main__":
    # 测试两种可视化方式
    base_pattern = "./augmented_dataset/image_{:06d}"
    start_idx = 39
    
    print("显示原始图片：")
    visualize_multiple_raw_images(base_pattern, start_idx)
    
    print("\n显示带标注的图片：")
    visualize_multiple_images(base_pattern, start_idx)
