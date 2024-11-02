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

def visualize_yolo_annotations(image_path: str, label_path: str):
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
        label_w, label_h = draw.textsize(label, font=font)
        draw.rectangle([x1, y1-label_h-4, x1+label_w+4, y1], 
                      fill=color)
        
        # 添加类别标签
        draw.text((x1+2, y1-label_h-2), label, 
                 fill=(255, 255, 255),  # 白色文字
                 font=font)
    
    # 显示图像
    plt.figure(figsize=(12, 12))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.title('YOLO Annotations Visualization (Numbers and Letters)')
    plt.show()

if __name__ == "__main__":
    # 选择一张图片进行测试
    base_path = "./augmented_dataset/image_000119"  # 替换为实际的图片基础路径
    image_path = f"{base_path}.png"  # 图片路径
    label_path = f"{base_path}.txt"  # 对应的标注文件
    
    visualize_yolo_annotations(image_path, label_path) 
