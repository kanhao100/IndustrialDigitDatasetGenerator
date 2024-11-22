import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
import sys
from typing import List, Tuple

from default_config import (FONT_DIRECTORIES, EXCLUDED_FONTS, EXCLUDED_KEYWORDS,
                   WINDOWS_FONTS, IMAGE_SIZE, TEXT_COLOR, OUTPUT_BASE_DIR,
                   DEFAULT_CHARS)

def trim_image(image):
    """裁剪图像，去除周围的空白"""
    # 获取图像的边界框
    bg = Image.new(image.mode, image.size, 'white')
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    
    if bbox:
        # 在边界框周围添加小边距（2像素）
        padding = 2
        bbox = (max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(image.size[0], bbox[2] + padding),
                min(image.size[1], bbox[3] + padding))
        return image.crop(bbox)
    return image

def extract_chars_from_ttf(ttf_path: str, 
                         output_base_dir: str,
                         chars_to_extract: List[str] = None,
                         font_size: int = 48,
                         image_size: Tuple[int, int] = (64, 64),
                         text_color: Tuple[int, int, int] = (0, 0, 0)) -> bool:
    """从TTF文件中提取指定字符
    Args:
        ttf_path: TTF字体文件路径
        output_base_dir: 输出目录基础路径
        chars_to_extract: 要提取的字符列表，默认为数字0-9
        font_size: 字体大小，默认48
        image_size: 输出图像尺寸，默认(64, 64)
        text_color: 文字颜色，默认黑色(0, 0, 0)
    Returns:
        bool: 是否成功提取
    """
    # 默认提取数字0-9
    if chars_to_extract is None:
        chars_to_extract = [str(i) for i in range(10)]
    
    # 创建所有需要的子文件夹
    for char in chars_to_extract:
        # 对于字母，使用"upper_X"和"lower_x"格式的文件夹
        if char.isalpha():
            if char.isupper():
                folder_name = f"upper_{char}"
            else:
                folder_name = f"lower_{char}"
        else:
            folder_name = char
        # 1. Windows：默认情况下，Windows文件系统（如NTFS）是不区分大小写的。这意味着在Windows上，"A"和"a"会被视为相同的文件夹名称。
        # 2. macOS：默认情况下，macOS的HFS+文件系统也是不区分大小写的，但可以配置为区分大小写。
        # 3. Linux：大多数Linux文件系统（如ext4）是区分大小写的。因此，在Linux上，"A"和"a"会被视为不同的文件夹。
        char_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(char_dir, exist_ok=True)

    # 获取字体文件名(不含扩展名)
    font_name = os.path.splitext(os.path.basename(ttf_path))[0]
    
    try:
        # 使用PIL创建字符图像
        font = ImageFont.truetype(ttf_path, font_size)
        
        for char in chars_to_extract:
            # 创建新图像（使用RGBA模式支持透明背景）
            img = Image.new('RGBA', IMAGE_SIZE, (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # 获取字体的完整度量信息
            ascent, descent = font.getmetrics()
            total_height = ascent + descent
            
            # 获取文字大小
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            
            # 计算居中位置
            x = (IMAGE_SIZE[0] - text_width) // 2
            # 考虑字体的上升和下降部分来计算垂直居中位置
            y = (IMAGE_SIZE[1] - total_height) // 2
            
            # 绘制文字，加上ascent偏移以确保完整显示
            draw.text((x + int(text_width / 2), y + ascent), char, font=font, fill=TEXT_COLOR, anchor="ms")
            
            # 裁剪图像
            trimmed_img = trim_image(img)
            
            # 生成输出文件名
            # 对于字母，使用"upper_X"和"lower_x"格式的文件夹
            if char.isalpha():
                if char.isupper():
                    folder_name = f"upper_{char}"
                    safe_char = f"upper_{char}"
                else:
                    folder_name = f"lower_{char}"
                    safe_char = f"lower_{char}"
            else:
                folder_name = char
                safe_char = "".join(c if c.isalnum() else f"_{ord(c)}_" for c in char) 
                # 处理文件名中的特殊字符，确保生成的文件名在所有操作系统中都是合法的
            
            output_path = os.path.join(output_base_dir, folder_name, f"{font_name}_{safe_char}.png")
            
            # 保存图像
            trimmed_img.save(output_path, "PNG")
            
        print(f"成功从 {ttf_path} 提取字符")
        return True
        
    except Exception as e:
        print(f"处理字体文件时出错: {str(e)}")
        return False

def scan_font_directory(directory):
    """扫描目录及其子目录中的所有字体文件"""
    font_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                full_path = os.path.join(root, file)
                font_name = os.path.splitext(file)[0]
                font_files.append((font_name, full_path))
    return font_files

def should_exclude_font(font_name):
    """检查字体是否应该被排除"""
    # 将字体名转换为小写进行比较
    font_name_lower = font_name.lower()
    
    # 检查完整名称（不区分大小写）
    if font_name_lower in {f.lower() for f in EXCLUDED_FONTS}:
        return True
    
    # 检查是否包含某些关键字
    for keyword in EXCLUDED_KEYWORDS:
        if keyword.lower() in font_name_lower:
            return True
    
    return False

def main():
    # 创建输出根目录
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)
    
    processed_count = 0
    
    # 处理 Windows 预定义字体列表
    for font_name, font_files in WINDOWS_FONTS.items():
        # 检查字体名是否在排除列表中
        if should_exclude_font(font_name):
            print(f"Skipping excluded font: {font_name}")
            continue
            
        for font_file in font_files:
            # 检查文件名是否在排除列表中
            if should_exclude_font(os.path.splitext(font_file)[0]):
                print(f"Skipping excluded font file: {font_file}")
                continue
                
            # 在所有字体目录中查找字体文件
            found = False
            for fonts_dir in FONT_DIRECTORIES:
                font_path = os.path.join(fonts_dir, font_file)
                if os.path.exists(font_path):
                    print(f"Processing predefined font: {font_name} - {font_file}")
                    if extract_chars_from_ttf(font_path, OUTPUT_BASE_DIR, DEFAULT_CHARS):
                        processed_count += 1
                    found = True
                    break
            if not found:
                # print(f"未找到预定义字体文件: {font_file}")
                print(f"Font file not found: {font_file}")
    
    # 扫描所有字体目录中的其他字体文件
    processed_files = set()
    for fonts_dir in FONT_DIRECTORIES:
        if os.path.exists(fonts_dir):
            print(f"\nScanning directory: {fonts_dir}")
            additional_fonts = scan_font_directory(fonts_dir)
            
            for font_name, font_path in additional_fonts:
                # 检查是否应该排除该字体
                if should_exclude_font(font_name):
                    print(f"Skipping excluded font: {font_name}")
                    continue
                    
                # 检查是否已处理过该文件
                if font_path.lower() not in processed_files:
                    print(f"Processing additional font: {font_name}")
                    if extract_chars_from_ttf(font_path, OUTPUT_BASE_DIR, DEFAULT_CHARS):
                        processed_count += 1
                    processed_files.add(font_path.lower())
        else:
            print(f"Directory does not exist: {fonts_dir}")
    
    # print(f"\n处理完成！成功处理了 {processed_count} 个字体文件")
    print(f"\nProcessing completed! Successfully processed {processed_count} font files")

if __name__ == "__main__":
    main() 