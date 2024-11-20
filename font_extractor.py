import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
import sys
from typing import List, Tuple
import random
import numpy as np


from default_config import (FONT_DIRECTORIES, EXCLUDED_FONTS, EXCLUDED_KEYWORDS,
                   WINDOWS_FONTS, IMAGE_SIZE, TEXT_COLOR, OUTPUT_BASE_DIR,
                   DEFAULT_CHARS)

class FontExtractor:
    def __init__(self, font_directories=FONT_DIRECTORIES, 
                 excluded_fonts=EXCLUDED_FONTS,
                 excluded_keywords=EXCLUDED_KEYWORDS,
                 windows_fonts=WINDOWS_FONTS,
                 image_size=IMAGE_SIZE,
                 text_color=TEXT_COLOR,
                 verbose=True):
        self.font_directories = font_directories
        self.excluded_fonts = excluded_fonts
        self.excluded_keywords = excluded_keywords
        self.windows_fonts = windows_fonts
        self.image_size = image_size
        self.text_color = text_color
        self.verbose = verbose
        
        # 存储可用的字体路径
        self.available_fonts = {}  # 格式: {font_name: font_path}
        # 初始化扫描可用字体
        self._initialize_available_fonts()
        
    def _print_verbose(self, message):
        """控制是否显示处理信息"""
        if self.verbose:
            print(message)
            
    def _initialize_available_fonts(self):
        """初始化扫描并存储所有可用的字体路径"""
        processed_files = set()
        
        # 处理Windows预定义字体
        for font_name, font_files in self.windows_fonts.items():
            if self._should_exclude_font(font_name):
                self._print_verbose(f"跳过排除的字体: {font_name}")
                continue
                
            for font_file in font_files:
                if self._should_exclude_font(os.path.splitext(font_file)[0]):
                    continue
                    
                for fonts_dir in self.font_directories:
                    font_path = os.path.join(fonts_dir, font_file)
                    if os.path.exists(font_path):
                        try:
                            # 验证字体文件是否可用
                            ImageFont.truetype(font_path, 12)
                            self.available_fonts[font_name] = font_path
                            self._print_verbose(f"找到预定义字体: {font_name}")
                            break
                        except Exception:
                            continue
        
        # 扫描额外的字体文件
        for fonts_dir in self.font_directories:
            if os.path.exists(fonts_dir):
                self._print_verbose(f"\n扫描目录: {fonts_dir}")
                additional_fonts = self._scan_font_directory(fonts_dir)
                
                for font_name, font_path in additional_fonts:
                    if (not self._should_exclude_font(font_name) and 
                        font_path.lower() not in processed_files):
                        try:
                            # 验证字体文件是否可用
                            ImageFont.truetype(font_path, 12)
                            self.available_fonts[font_name] = font_path
                            self._print_verbose(f"找到额外字体: {font_name}")
                            processed_files.add(font_path.lower())
                        except Exception:
                            continue
                            
        self._print_verbose(f"\n共找到 {len(self.available_fonts)} 个可用字体")

    def get_font_image_size(self, font_size: int) -> Tuple[int, int]:
        """根据字体大小返回合适的图像尺寸"""
        # 图像尺寸应该是字体大小的倍数，确保有足够空间
        multiplier = 1.2  # 可据需要调整这个倍数
        size = int(font_size * multiplier)
        return (size, size)

    def extract_chars(self, chars_to_extract: List[str] = None,
                     font_size: int = 64) -> dict:
        """从所有可用字体中提取指定字符
        Args:
            chars_to_extract: 要提取的字符列表
            font_size: 字体大小
        Returns:
            dict: {folder_name: {font_name: PIL.Image}}
        """
        if chars_to_extract is None:
            chars_to_extract = [str(i) for i in range(10)]
            
        # 根据字体大小动态调整图像尺寸
        image_size = self.get_font_image_size(font_size)
        char_images = {}
        
        total_fonts = len(self.available_fonts)
        for idx, (font_name, font_path) in enumerate(self.available_fonts.items(), 1):
            self._print_verbose(f"处理字体 [{idx}/{total_fonts}]: {font_name}")
            
            try:
                font_chars = self._extract_chars_from_ttf(
                    font_path, 
                    chars_to_extract,
                    font_size,
                    image_size
                )
                if font_chars:
                    self._update_char_images(char_images, font_chars)
            except Exception as e:
                self._print_verbose(f"处理字体 {font_name} 时出错: {str(e)}")
                continue
                
        return char_images

    def _extract_chars_from_ttf(self, ttf_path: str,
                              chars_to_extract: List[str],
                              font_size: int,
                              image_size: Tuple[int, int]) -> dict:
        """从单个TTF文件中提取字符"""
        font_name = os.path.splitext(os.path.basename(ttf_path))[0]
        char_images = {}
        
        try:
            font = ImageFont.truetype(ttf_path, font_size)
            
            for char in chars_to_extract:
                img = Image.new('RGBA', image_size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(img)
                
                # 获取字体度量信息
                ascent, descent = font.getmetrics()
                total_height = ascent + descent
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                
                # 计算居中位置
                x = (image_size[0] - text_width) // 2
                y = (image_size[1] - total_height) // 2
                
                # 绘制文字
                draw.text((x + int(text_width / 2), y + ascent), char, 
                         font=font, fill=self.text_color, anchor="ms")
                
                trimmed_img = self._trim_image(img)
                
                # 确定文件夹名称
                folder_name = (f"upper_{char}" if char.isupper() else 
                             f"lower_{char}" if char.isalpha() else char)
                
                if folder_name not in char_images:
                    char_images[folder_name] = {}
                char_images[folder_name][font_name] = trimmed_img
            
            return char_images
            
        except Exception as e:
            self._print_verbose(f"处理字体文件时出错: {str(e)}")
            return None

    def _trim_image(self, image):
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

    def _update_char_images(self, new_images: dict):
        """更新字符图像字典"""
        for folder_name, font_images in new_images.items():
            if folder_name not in self.char_images:
                self.char_images[folder_name] = {}
            self.char_images[folder_name].update(font_images)

    def _should_exclude_font(self, font_name: str) -> bool:
        """检查字体是否应该被排除"""
        # 将字体名转换为小写进行比较
        font_name_lower = font_name.lower()
        
        # 检查完整名称（不区分大小写）
        if font_name_lower in {f.lower() for f in self.excluded_fonts}:
            return True
        
        # 检查是否包含某些关键字
        for keyword in self.excluded_keywords:
            if keyword.lower() in font_name_lower:
                return True
        
        return False

    def _scan_font_directory(self, directory: str) -> List[Tuple[str, str]]:
        """扫描目录中的字体文件"""
        font_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                    full_path = os.path.join(root, file)
                    font_name = os.path.splitext(file)[0]
                    font_files.append((font_name, full_path))
        return font_files

    def extract_single_char(self, char: str, font_size: int, target_size: int = None) -> Tuple[str, np.ndarray]:
        """提取单个字符的图像，使用随机字体
        Args:
            char: 要提取的单个字符
            font_size: 字体大小
            target_size: 目标输出尺寸，如果指定则会调整到该尺寸
        Returns:
            Tuple[str, np.ndarray]: (字体名称, 字符图像数组)
        """
        # 随机选择一个可用字体
        font_name = random.choice(list(self.available_fonts.keys()))
        font_path = self.available_fonts[font_name]
        
        # 根据字体大小动态调整图像尺寸
        image_size = self.get_font_image_size(font_size)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
            
            # 创建新图像
            img = Image.new('RGBA', image_size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # 获取字体度量信息
            ascent, descent = font.getmetrics()
            total_height = ascent + descent
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            
            # 计算居中位置
            x = (image_size[0] - text_width) // 2
            y = (image_size[1] - total_height) // 2
            
            # 绘制文字
            draw.text((x + int(text_width / 2), y + ascent), char, 
                     font=font, fill=self.text_color, anchor="ms")
            
            # 裁剪图像
            # trimmed_img = self._trim_image(img)
            trimmed_img = img
            
            # 转换为灰度图
            gray_img = trimmed_img.convert('L')
            img_array = np.array(gray_img)
            
            # 如果指定了目标尺寸，调整图像大小
            if target_size is not None:
                img_array = self._resize_digit(img_array, target_size)
            
            return font_name, img_array
            
        except Exception as e:
            self._print_verbose(f"处理字体 {font_name} 时出错: {str(e)}")
            return None, None

    def _resize_digit(self, img_array: np.ndarray, target_size: int) -> np.ndarray:
        """调整数字图片大小
        Args:
            img_array: 输入图像数组，形状为(H, W)
            target_size: 目标尺寸
        Returns:
            np.ndarray: 调整大小后的图像数组，形状为(target_size, target_size)
        """
        # 计算有效边界框（非白色区域）
        y_indices, x_indices = np.where(img_array < 55)
        if len(y_indices) > 0 and len(x_indices) > 0:
            min_x = np.min(x_indices)
            max_x = np.max(x_indices)
            min_y = np.min(y_indices)
            max_y = np.max(y_indices)
            
            # 裁剪到有效区域
            img_array = img_array[min_y:max_y+1, min_x:max_x+1]
            
            # 确保裁剪后的图像至少是 1x1 的
            if img_array.shape[0] == 0 or img_array.shape[1] == 0:
                img_array = np.array([[0]], dtype=np.uint8)
        else:
            # 如果没有有效像素，返回一个小的黑色图像
            img_array = np.array([[0]], dtype=np.uint8)
        
        # # 直接将裁剪后的图像调整到目标大小，不保持宽高比
        # resized_img = Image.fromarray(img_array).resize(
        #     (target_size, target_size), 
        #     Image.LANCZOS
        # )
        # final_img = np.array(resized_img)
        # return final_img
        
        return img_array


def main():
    # 创建FontExtractor实例
    extractor = FontExtractor(verbose=True)

    # 提取单个字符
    char = '1'
    font_size = 800
    font_name, char_image = extractor.extract_single_char(char, font_size, target_size=800)

    if char_image is not None:
        # 保存图像到文件
        Image.fromarray(np.array(char_image)).save(f"{char}.png")
        print(f"成功使用字体 {font_name} 提取字符 {char} 并保存为 {char}.png")
    else:
        print("提取字符失败")
            

if __name__ == "__main__":
    main() 