import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
import sys

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

def extract_numbers_from_ttf(ttf_path, output_base_dir):
    """提取单个字体文件中的数字"""
    # 获取字体文件名(不含扩展名)
    font_name = os.path.splitext(os.path.basename(ttf_path))[0]
    
    try:
        # 使用PIL创建数字图像
        font = ImageFont.truetype(ttf_path, 48)
        
        for digit in range(10):
            # 创建新（使用RGBA模式支持透明背景）
            img = Image.new('RGBA', IMAGE_SIZE, (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # 取文字大小
            text = str(digit)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算居中位置
            x = (IMAGE_SIZE[0] - text_width) // 2
            y = (IMAGE_SIZE[1] - text_height) // 2
            
            # 绘制文字
            draw.text((x, y), text, font=font, fill=TEXT_COLOR)
            
            # 裁剪图像
            trimmed_img = trim_image(img)
            
            # 保存图像到对应数字文件夹
            output_path = os.path.join(output_base_dir, str(digit), f"{font_name}_{digit}.png")
            trimmed_img.save(output_path, "PNG")
            
        print(f"成功从 {ttf_path} 提取数字")
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
    
    # 创建0-9的子文件夹
    for i in range(10):
        digit_dir = os.path.join(output_base_dir, str(i))
        if not os.path.exists(digit_dir):
            os.makedirs(digit_dir)
    
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
                    if extract_numbers_from_ttf(font_path, output_base_dir):
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
                    if extract_numbers_from_ttf(font_path, output_base_dir):
                        processed_count += 1
                    processed_files.add(font_path.lower())
        else:
            print(f"目录不存在: {fonts_dir}")
    
    print(f"\n处理完成！成功处理了 {processed_count} 个字体文件")

if __name__ == "__main__":
    main() 