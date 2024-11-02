import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def visualize_yolo_annotations(image_path: str, label_path: str):
    """可视化YOLO格式的标注"""
    # 读取图像
    img = Image.open(image_path)
    width, height = img.size
    
    # 转换为RGB以便绘制彩色标注
    vis_img = img.convert('RGB')
    draw = ImageDraw.Draw(vis_img)
    
    # 读取标注文件
    with open(label_path, 'r') as f:
        annotations = f.readlines()
    
    # 为每个数字类别分配不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))  # 10种颜色对应0-9
    colors = (colors[:, :3] * 255).astype(np.uint8)  # 转换为RGB格式
    
    # 绘制每个标注框
    for ann in annotations:
        # 解析YOLO格式的标注：class_id center_x center_y width height
        class_id, center_x, center_y, w, h = map(float, ann.strip().split())
        class_id = int(class_id)
        
        # 转换为像素坐标
        center_x = int(center_x * width)
        center_y = int(center_y * height)
        box_w = int(w * width)
        box_h = int(h * height)
        
        # 计算框的左上角和右下角坐标
        x1 = int(center_x - box_w/2)
        y1 = int(center_y - box_h/2)
        x2 = int(center_x + box_w/2)
        y2 = int(center_y + box_h/2)
        
        # 获取当前类别的颜色
        color = tuple(map(int, colors[class_id]))
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 添加类别标签
        label = str(class_id)
        draw.text((x1, y1-15), label, fill=color)
    
    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.title('YOLO Annotations Visualization')
    plt.show()

if __name__ == "__main__":
    # 选择一张图片进行测试
    base_path = "./augmented_dataset/image_000001"  # 替换为实际的图片基础路径
    image_path = f"{base_path}.png"  # 图片路径
    label_path = f"{base_path}.txt"  # 对应的标注文件
    
    visualize_yolo_annotations(image_path, label_path) 