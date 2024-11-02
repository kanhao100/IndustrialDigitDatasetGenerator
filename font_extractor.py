import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
import sys
from typing import List, Tuple

# 全局常量定义
IMAGE_SIZE = (64, 64)  # 输出图像大小
TEXT_COLOR = 'black'   # 文字颜色

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
            img = Image.new('RGBA', image_size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # 获取字体的完整度量信息
            ascent, descent = font.getmetrics()
            total_height = ascent + descent
            
            # 获取文字大小
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            
            # 计算居中位置
            x = (image_size[0] - text_width) // 2
            # 考虑字体的上升和下降部分来计算垂直居中位置
            y = (image_size[1] - total_height) // 2
            
            # 绘制文字，加上ascent偏移以确保完整显示
            draw.text((x + int(text_width / 2), y + ascent), char, font=font, fill=text_color, anchor="ms")
            
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
    # 需要排除的字体列表（不区分大小写）
    excluded_fonts = {
        'bssym7',
        'holomdl2',
        'marlett',
        'inkfree',
        'javatext',
        'mtextra',
        'refspcl',
        'segmdl2',
        'segoepr',
        'segoeprb',
        'segoesc',
        'segoescb',
        'stcaiyun',
        'sthupo',
        'symbol',
        'webdings',
        'wingding',
        'wingdng2',
        'BRADHITC',
        'ITCKRIST',
        'MISTRAL',
        'mvboli',
        'PAPYRUS',
        'PRISTINA',
        'FREESCPT'
    }
    
    # 将字体名转换为小写进行比较
    font_name_lower = font_name.lower()
    
    # 检查完整名称（不区分大小写）
    for excluded_font in excluded_fonts:
        if font_name_lower == excluded_font.lower():
            return True
    
    # 检查是否包含某些关键字（部分匹配，不区分大小写）
    excluded_keywords = {'symbol', 'wing', 'webding'}
    for keyword in excluded_keywords:
        if keyword.lower() in font_name_lower:
            return True
    
    return False

def main():
    # Windows常用字体列表（包括变体）
    windows_fonts = {
        # 英文字体及其变体
        'Arial': [
            'arial.ttf',          # Regular
            'arialbd.ttf',        # Bold
            'ariali.ttf',         # Italic
            'arialbi.ttf',        # Bold Italic
            'arialnb.ttf',        # Narrow Bold
            'arialni.ttf',        # Narrow Italic
            'arialnbi.ttf',       # Narrow Bold Italic
            'arialn.ttf',         # Narrow
        ],
        
        'Times New Roman': [
            'times.ttf',          # Regular
            'timesbd.ttf',        # Bold
            'timesi.ttf',         # Italic
            'timesbi.ttf',        # Bold Italic
        ],
        
        'Calibri': [
            'calibri.ttf',        # Regular
            'calibrib.ttf',       # Bold
            'calibrii.ttf',       # Italic
            'calibriz.ttf',       # Bold Italic
            'calibril.ttf',       # Light
            'calibrili.ttf',      # Light Italic
        ],
        
        'Tahoma': [
            'tahoma.ttf',         # Regular
            'tahomabd.ttf',       # Bold
        ],
        
        'Verdana': [
            'verdana.ttf',        # Regular
            'verdanab.ttf',       # Bold
            'verdanai.ttf',       # Italic
            'verdanaz.ttf',       # Bold Italic
        ],
        
        # 中文字体及其变体
        'Microsoft YaHei': [
            'msyh.ttc',           # Regular
            'msyhbd.ttc',         # Bold
            'msyhl.ttc',          # Light
            'msyhsb.ttc',         # Semi-bold
        ],
        
        'SimSun': [
            'simsun.ttc',         # Regular
            'simsunb.ttf',        # Bold
            'SURSONG.TTF',        # 宋体-超细
            'FZSTK.TTF',          # 宋体-特细
        ],
        
        'Microsoft JhengHei': [
            'msjh.ttc',           # Regular
            'msjhbd.ttc',         # Bold
            'msjhl.ttc',          # Light
        ],
        
        'KaiTi': [
            'simkai.ttf',         # Regular
            'STKAITI.TTF',        # 华文楷体
        ],
        
        'FangSong': [
            'simfang.ttf',        # Regular
            'STFANGSO.TTF',       # 华文仿宋
        ],
        
        'SimHei': [
            'simhei.ttf',         # Regular
        ],
        
        'DengXian': [
            'dengxian.ttf',       # Regular
            'dengxianb.ttf',      # Bold
            'dengxianl.ttf',      # Light
        ],
        
        'SourceHanSans': [
            'SourceHanSansSC-Regular.otf',    # Regular
            'SourceHanSansSC-Bold.otf',       # Bold
            'SourceHanSansSC-Heavy.otf',      # Heavy
            'SourceHanSansSC-Light.otf',      # Light
            'SourceHanSansSC-Medium.otf',     # Medium
            'SourceHanSansSC-Normal.otf',     # Normal
        ],
        
        'NSimSun': [
            'nsimsun.ttf',        # Regular
        ],
        
        'Microsoft NeoGothic': [
            'msneo.ttc',          # Regular
            'msneob.ttc',         # Bold
        ],
        
        'FangSong_GB2312': [
            'SIMFANG.TTF',        # Regular
        ],
        
        'YouYuan': [
            'SIMYOU.TTF',         # Regular
        ],
        
        'LiSu': [
            'SIMLI.TTF',          # Regular
        ],
        
        'STXihei': [
            'STXIHEI.TTF',        # Regular
        ],
        
        'STKaiti': [
            'STKAITI.TTF',        # Regular
        ],
        
        'STFangsong': [
            'STFANGSO.TTF',       # Regular
        ],
        
        'STSong': [
            'STSONG.TTF',         # Regular
        ],
        
        'STZhongsong': [
            'STZHONGS.TTF',       # Regular
        ],
    }
    
    # 字体目录列表
    font_directories = [
        "C:\\Windows\\Fonts",
        "C:\\Program Files\\Microsoft Office\\root\\vfs\\Fonts",
        "C:\\Program Files (x86)\\Microsoft Office\\root\\vfs\\Fonts",
        "C:\\Program Files\\Common Files\\Microsoft Shared\\Fonts",
        "C:\\Program Files (x86)\\Common Files\\Microsoft Shared\\Fonts",
    ]
    
    # 创建输出根目录
    output_base_dir = "font_numbers"
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # chars = [str(i) for i in range(10)]
    chars = [str(i) for i in range(10)] + list("cmCM")

    processed_count = 0
    
    # 处理 Windows 预定义字体列表
    for font_name, font_files in windows_fonts.items():
        # 检查字体名是否在排除列表中
        if should_exclude_font(font_name):
            print(f"跳过排除的字体: {font_name}")
            continue
            
        for font_file in font_files:
            # 检查文件名是否在排除列表中
            if should_exclude_font(os.path.splitext(font_file)[0]):
                print(f"跳过排除的字体文件: {font_file}")
                continue
                
            # 在所有字体目录中查找字体文件
            found = False
            for fonts_dir in font_directories:
                font_path = os.path.join(fonts_dir, font_file)
                if os.path.exists(font_path):
                    print(f"正在处理预定义字体: {font_name} - {font_file}")
                    if extract_chars_from_ttf(font_path, output_base_dir, chars):
                        processed_count += 1
                    found = True
                    break
            if not found:
                print(f"未找到预定义字体文件: {font_file}")
    
    # 扫描所有字体目录中的其他字体文件
    processed_files = set()
    for fonts_dir in font_directories:
        if os.path.exists(fonts_dir):
            print(f"\n扫描目录: {fonts_dir}")
            additional_fonts = scan_font_directory(fonts_dir)
            
            for font_name, font_path in additional_fonts:
                # 检查是否应该排除该字体
                if should_exclude_font(font_name):
                    print(f"跳过排除的字体: {font_name}")
                    continue
                    
                # 检查是否已处理过该文件
                if font_path.lower() not in processed_files:
                    print(f"正在处理额外字体: {font_name}")
                    if extract_chars_from_ttf(font_path, output_base_dir, chars):
                        processed_count += 1
                    processed_files.add(font_path.lower())
        else:
            print(f"目录不存在: {fonts_dir}")
    
    print(f"\n处理完成！成功处理了 {processed_count} 个字体文件")

if __name__ == "__main__":
    main() 