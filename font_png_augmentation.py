import os
import numpy as np
from PIL import Image, ImageDraw
import random
from typing import Tuple, List, Dict
from multiprocessing import Pool
import multiprocessing
import math

class ImageAugmentor:
    def __init__(
        self,
        canvas_size: int = 256,          # 输出图像的尺寸大小，生成 canvas_size x canvas_size 的正方形图像
        background_noise_type: str = "perlin",  # 背景噪声类型：
                                           # - 'perlin': 柏林噪声，生成连续的、自然的纹理
                                           # - 'simplex': 单纯形噪声，类似柏林噪声但性能更好
                                           # - 'gaussian': 高斯噪声，完全随机的噪点
        background_noise_intensity: float = 0.1,  # 背景噪声强度，范围 0.0-1.0
                                            # 值越大，背景噪声越明显
        digit_noise_intensity_range: Tuple[float, float] = (0.0, 0.2),  # 数字噪声强度范围，范围 0.0-1.0
        min_digits: int = 3,                   # 每张图像最少数字数量
        max_digits: int = 6,                   # 每张图像最多数字数量
        min_scale: float = 0.2,                # 数字最小缩放比例（相对于 canvas_size）
                                          # 例如：0.2 表示数字最小为画布的 20%
        max_scale: float = 0.4,                # 数字最大缩放比例（相对于 canvas_size）
                                          # 例如：0.4 表示数字最大为画布的 40%
        min_spacing: int = 5,                  # 数字之间的最小间距（像素）
        max_placement_attempts: int = 100,      # 寻找有效放置位置的最大尝试次数
                                          # 超过此次数认为无法放置更多数字
        use_real_background: bool = False,      # 新增：是否使用真实背景图
        real_background_dir: str = "./NEU-DET/IMAGES",  # 新增：真实背景图目录
        augmentation_types: List[str] = None,  # 新增：指定要使用的增强类型
        noise_types: List[str] = None,  # 新增：指定要使用的噪声类型
        occlusion_prob: float = 0.3,  # 新增：遮挡概率
        distortion_range: Tuple[float, float] = (0.8, 1.2),  # 新增：变形范围
        brightness_range: Tuple[float, float] = (0.7, 1.3),  # 新增：亮度调节范围
        noise_patterns: List[str] = None,
        noise_pattern_weights: Dict[str, float] = None,
        annotate_letters: bool = False,  # 新增：是否为字母生成YOLO标注
        letter_count: int = 999  # 字母出现次数限制
        
    ):
        self.canvas_size = canvas_size
        self.background_noise_type = background_noise_type
        self.background_noise_intensity = background_noise_intensity
        self.digit_noise_intensity_range = digit_noise_intensity_range
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_spacing = min_spacing
        self.max_placement_attempts = max_placement_attempts
        self.use_real_background = use_real_background
        self.real_background_dir = real_background_dir
        self.real_background_files = None
        self.augmentation_types = augmentation_types or [
            'noise', 'occlusion', 'distortion', 'aspect_ratio', 'rotation', 'brightness'
        ]
        self.noise_types = noise_types or [
            'gaussian', 'salt_pepper', 'speckle', 'poisson'
        ]
        self.occlusion_prob = occlusion_prob
        self.distortion_range = distortion_range
        self.brightness_range = brightness_range
        # 更新噪声图案参数
        self.noise_patterns = noise_patterns or [
            'circle', 'vertical_stripe', 'horizontal_stripe', 
            'rectangle', 'hexagon', 'triangle'
        ]
        self.noise_pattern_weights = noise_pattern_weights or {
            'circle': 0.2,
            'vertical_stripe': 0.2,
            'horizontal_stripe': 0.2,
            'rectangle': 0.2,
            'hexagon': 0.1,
            'triangle': 0.1
        }
        self.annotate_letters = annotate_letters
        self.letter_count = letter_count
        self.total_letters = 0  # 用于跟踪字母总数
        
        if use_real_background:
            self.real_background_files = [
                f for f in os.listdir(real_background_dir) 
                if f.endswith('.jpg') and not f.startswith('patches') 
                # 对于NEU-DET数据集，只使用jpg格式的背景图, 且排除掉patches开头的文件，因为遮挡过于严重
            ]
        
    def _generate_perlin_noise(self, shape: Tuple[int, int]) -> np.ndarray:
        """生成柏林噪声作为工业环境背景"""
        def perlin(x, y, seed=0):
            # 简化版柏林噪声实现
            np.random.seed(seed)
            p = np.arange(256, dtype=int)
            np.random.shuffle(p)
            p = np.stack([p, p]).flatten()
            
            xi = x.astype(int) & 255
            yi = y.astype(int) & 255
            
            g00 = p[p[xi] + yi]
            g10 = p[p[xi + 1] + yi]
            g01 = p[p[xi] + yi + 1]
            g11 = p[p[xi + 1] + yi + 1]
            
            # 线性插值
            xf = x - x.astype(int)
            yf = y - y.astype(int)
            
            u = xf * xf * (3.0 - 2.0 * xf)
            v = yf * yf * (3.0 - 2.0 * yf)
            
            return (g00 * (1-u) * (1-v) + 
                   g10 * u * (1-v) + 
                   g01 * (1-u) * v + 
                   g11 * u * v)

        x = np.linspace(0, 8, shape[0], endpoint=False)
        y = np.linspace(0, 8, shape[1], endpoint=False)
        x_grid, y_grid = np.meshgrid(x, y)
        
        noise = perlin(x_grid, y_grid)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return (noise * 255 * self.background_noise_intensity).astype(np.uint8)

    def _add_industrial_background(self, image: np.ndarray) -> np.ndarray:
        """添加工业环境背景"""
        if self.use_real_background and self.real_background_files:
            # 随机选择一张背景图
            bg_file = random.choice(self.real_background_files)
            bg_path = os.path.join(self.real_background_dir, bg_file)
            
            # 读取并调整背景图大小
            bg_img = Image.open(bg_path).convert('L')
            bg_img = bg_img.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)
            bg_array = np.array(bg_img)
            
            # 将背景图的亮度调整到合适范围
            bg_array = np.clip(bg_array * self.background_noise_intensity, 0, 255).astype(np.uint8)
            
            # 将原图叠加到背景上
            return np.clip(image.astype(float) - bg_array, 0, 255).astype(np.uint8)
        else:
            # 原有的背景生成逻辑
            if self.background_noise_type == "perlin":
                noise = self._generate_perlin_noise(image.shape)
            else:  # fallback to gaussian
                noise = np.random.normal(0, self.background_noise_intensity * 255, image.shape)
                
            # 添加一些随机的条纹和污点
            num_streaks = random.randint(2, 5)
            for _ in range(num_streaks):
                start_x = random.randint(0, image.shape[1])
                start_y = random.randint(0, image.shape[0])
                length = random.randint(20, 100)
                width = random.randint(1, 3)
                intensity = random.randint(5, 20)
                
                for i in range(length):
                    x = min(start_x + i, image.shape[1]-1)
                    for w in range(width):
                        y = min(start_y + w, image.shape[0]-1)
                        noise[y, x] = max(noise[y, x], intensity)

            return np.clip(image.astype(float) - noise, 0, 255).astype(np.uint8)

    def _add_digit_noise(self, digit_img: np.ndarray) -> np.ndarray:
        """为单个数字添加噪声，随机选择一种噪声类型"""
        # 创建非白色像素的掩码
        mask = digit_img < 255
        result = digit_img.copy()
        
        # 随机选择一种噪声类型
        noise_type = random.choice(self.noise_types)
        intensity = random.uniform(*self.digit_noise_intensity_range)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, digit_img.shape)
        elif noise_type == 'salt_pepper':
            noise = np.zeros_like(digit_img)
            salt = np.random.random(digit_img.shape) < intensity/2
            pepper = np.random.random(digit_img.shape) < intensity/2
            noise[salt] = 255
            noise[pepper] = 0
        elif noise_type == 'speckle':
            noise = intensity * digit_img * np.random.randn(*digit_img.shape)
        elif noise_type == 'poisson':
            noise = np.random.poisson(digit_img * intensity) - digit_img
            
        # 只对非白色区域添加噪声
        result[mask] = np.clip(result[mask] + noise[mask], 0, 255)
            
        return result.astype(np.uint8)

    def _apply_augmentations(self, digit_img: np.ndarray) -> np.ndarray:
        """应用多种数据增强"""
        # 将numpy数组转换为PIL图像以便处理
        img = Image.fromarray(digit_img)
        
        # 随机选择要应用的增强类型
        augmentations = random.sample(self.augmentation_types, 
                                    random.randint(1, len(self.augmentation_types)))
        
        for aug_type in augmentations:
            if aug_type == 'noise':
                # 噪声处理直接使用numpy数组
                img = np.array(img)
                img = self._add_digit_noise(img)
                img = Image.fromarray(img)
            
            elif aug_type == 'occlusion':
                if random.random() < self.occlusion_prob:
                    w, h = img.size
                    # 随机选择遮挡类型
                    occlusion_type = random.choice(['rectangle', 'vertical', 'horizontal', 'circle'])
                    
                    if occlusion_type == 'rectangle':
                        # 原有的矩形遮挡
                        occlude_w = int(w * random.uniform(0.1, 0.4))
                        occlude_h = int(h * random.uniform(0.1, 0.4))
                        x = random.randint(0, w - occlude_w)
                        y = random.randint(0, h - occlude_h)
                        occlude = Image.new('L', (occlude_w, occlude_h), 255)
                        img.paste(occlude, (x, y))
                    
                    elif occlusion_type == 'vertical':
                        # 竖条纹遮挡
                        num_stripes = random.randint(1, 2)  # 条纹数量
                        stripe_width = int(w * random.uniform(0.05, 0.15))  # 条纹宽度
                        for _ in range(num_stripes):
                            x = random.randint(0, w - stripe_width)
                            # 添加随机高度
                            stripe_height = int(h * random.uniform(0.4, 0.9))  # 条纹高度
                            y = random.randint(0, h - stripe_height)  # 随机起始位置
                            occlude = Image.new('L', (stripe_width, stripe_height), 255)
                            img.paste(occlude, (x, y))
                    
                    elif occlusion_type == 'horizontal':
                        # 横条纹遮挡
                        num_stripes = random.randint(1, 2)  # 条纹数量
                        stripe_height = int(h * random.uniform(0.05, 0.15))  # 条纹高度
                        for _ in range(num_stripes):
                            y = random.randint(0, h - stripe_height)
                            # 添加随机宽度
                            stripe_width = int(w * random.uniform(0.4, 0.9))  # 条纹宽度
                            x = random.randint(0, w - stripe_width)  # 随机起始位置
                            occlude = Image.new('L', (stripe_width, stripe_height), 255)
                            img.paste(occlude, (x, y))
                    
                    elif occlusion_type == 'circle':
                        # 圆形遮挡
                        from PIL import ImageDraw
                        num_circles = random.randint(1, 2)  # 圆形数
                        for _ in range(num_circles):
                            # 创建一个透明遮罩
                            mask = Image.new('L', (w, h), 0)
                            draw = ImageDraw.Draw(mask)
                            
                            # 随机圆形参数
                            radius = int(min(w, h) * random.uniform(0.1, 0.2))
                            center_x = random.randint(radius, w - radius)
                            center_y = random.randint(radius, h - radius)
                            
                            # 绘制白色圆形
                            draw.ellipse(
                                [center_x - radius, center_y - radius,
                                 center_x + radius, center_y + radius],
                                fill=255
                            )
                            # 将圆形遮罩应用到图像上
                            img.paste(255, (0, 0), mask)
            
            elif aug_type == 'distortion':
                # 使用PIL的变形方法
                w, h = img.size
                # 限制扭曲范围在更小的区间内，避免过度变形
                max_distort = 0.15  # 最大扭曲幅度为15%
                
                # 随机生成四个角点的偏移量
                offsets = [
                    (random.uniform(-max_distort, max_distort) * w,  # 左上x,y
                     random.uniform(-max_distort, max_distort) * h),
                    (random.uniform(-max_distort, max_distort) * w,  # 右上x,y
                     random.uniform(-max_distort, max_distort) * h),
                    (random.uniform(-max_distort, max_distort) * w,  # 右下x,y
                     random.uniform(-max_distort, max_distort) * h),
                    (random.uniform(-max_distort, max_distort) * w,  # 左下x,y
                     random.uniform(-max_distort, max_distort) * h),
                ]
                
                # 计算变换后的四个角点坐标
                width, height = w - 1, h - 1
                coeffs = (
                    0 + offsets[0][0], 0 + offsets[0][1],           # 左上角
                    width + offsets[1][0], 0 + offsets[1][1],       # 右上角
                    width + offsets[2][0], height + offsets[2][1],  # 右下角
                    0 + offsets[3][0], height + offsets[3][1]       # 左下角
                )
                
                img = img.transform(
                    (w, h),
                    Image.QUAD,
                    coeffs,
                    Image.BICUBIC,
                    fillcolor=255
                )
            
            elif aug_type == 'aspect_ratio':
                # 随机改变长宽比
                w, h = img.size
                new_w = int(w * random.uniform(0.8, 1.2))
                new_h = int(h * random.uniform(0.8, 1.2))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                
                # 创建新的白色背景图像
                new_img = Image.new('L', (w, h), 255)
                # 计算粘贴位置使图像居中
                paste_x = (w - new_w) // 2
                paste_y = (h - new_h) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img
            
            elif aug_type == 'rotation':
                # 随机旋转
                angle = random.uniform(-10, 10)
                img = img.rotate(angle, Image.BICUBIC, expand=False, fillcolor=255)
            
            elif aug_type == 'brightness':
                # 随机调节亮度
                img_array = np.array(img)
                # 识别非背景区域（灰度值小于240的区域视为数字）
                mask = img_array < 240
                
                if mask.any():
                    brightness_factor = random.uniform(*self.brightness_range)
                    adjusted = img_array.copy()
                    
                    if brightness_factor < 1:  # 变暗
                        # 将非背景区域的像素值向黑色(0)靠近
                        adjusted[mask] = (img_array[mask] * brightness_factor).astype(np.uint8)
                    else:  # 变亮
                        # 将非背景区域的像素值向背景色(255)靠近
                        diff = 255 - img_array[mask]  # 与背景色的差值
                        adjusted[mask] = (
                            img_array[mask] + diff * (brightness_factor - 1)
                        ).astype(np.uint8)
                    
                    # 确保值在有效范围内
                    adjusted = np.clip(adjusted, 0, 255)
                    img = Image.fromarray(adjusted)
        
        # 最后将PIL图像转回numpy数组
        return np.array(img)

    # def _load_digit(self, digit_path: str) -> np.ndarray:
    #     """加载数字图片"""
    #     return np.array(Image.open(digit_path).convert('L'))
        
    def _load_digit(self, digit_path: str) -> np.ndarray:
        """加载数字的-1通道,透明度数据(alpha 通道),并反转黑白"""
        img = Image.open(digit_path).split()[-1].convert('L')
        img_array = np.array(img)
        inverted_array = 255 - img_array
        return inverted_array
        
    def _resize_digit(self, img_array: np.ndarray, target_size: int) -> np.ndarray:
        """调整数字图片大小
        Args:
            img_array: 输入图像数组，形状为(H, W)
            target_size: 目标尺寸
        Returns:
            np.ndarray: 调整大小后的图像数组，形状为(target_size, target_size)
        """
        # 计算有效边界框（非白色区域）
        y_indices, x_indices = np.where(img_array < 255)
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
        
        # 直接将裁剪后的图像调整到目标大小，不保持宽高比
        resized_img = Image.fromarray(img_array).resize((target_size, target_size), Image.LANCZOS)
        final_img = np.array(resized_img)
        
        return final_img

    def _get_bounding_box(self, digit_img: np.ndarray) -> Tuple[int, int, int, int]:
        """获取数字图像中非空白像素的边界框
        返回：(min_x, min_y, width, height)
        """
        # 因为是灰度图，背景是255（白色），我们找出非255的像素
        y_indices, x_indices = np.where(digit_img < 255)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            return (0, 0, digit_img.shape[1], digit_img.shape[0])
            
        min_x = np.min(x_indices)
        max_x = np.max(x_indices)
        min_y = np.min(y_indices)
        max_y = np.max(y_indices)
        
        return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)

    def _find_valid_position(
        self, 
        canvas: np.ndarray, 
        digit_img: np.ndarray,
        occupied_positions: List[Tuple[int, int, np.ndarray]]
    ) -> Tuple[int, int]:
        """找到一个有效的放置位置"""
        digit_size = digit_img.shape[0]
        # 获取当前数字的实际边界框
        curr_bbox = self._get_bounding_box(digit_img)
        
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(0, self.canvas_size - digit_size)
            y = random.randint(0, self.canvas_size - digit_size)
            
            # 检查是否与已有数字重叠
            valid = True
            for ox, oy, other_img in occupied_positions:
                # 获取已放置数字的边界框
                other_bbox = self._get_bounding_box(other_img)
                
                # 计算当前数字实际位置的边界框
                curr_x = x + curr_bbox[0]
                curr_y = y + curr_bbox[1]
                curr_w = curr_bbox[2]
                curr_h = curr_bbox[3]
                
                # 计算已放置数字实际位置的界框
                other_x = ox + other_bbox[0]
                other_y = oy + other_bbox[1]
                other_w = other_bbox[2]
                other_h = other_bbox[3]
                
                # 检查实际边界框是否重叠（考虑间距）
                if (curr_x < other_x + other_w + self.min_spacing and
                    curr_x + curr_w + self.min_spacing > other_x and
                    curr_y < other_y + other_h + self.min_spacing and
                    curr_y + curr_h + self.min_spacing > other_y):
                    valid = False
                    break
            
            if valid:
                return x, y
                
        raise ValueError("无法找到有效的放置位置")

    def _generate_noise_pattern(self, size: int) -> np.ndarray:
        """生成各种噪声图案"""
        pattern_type = random.choices(
            list(self.noise_pattern_weights.keys()),
            weights=list(self.noise_pattern_weights.values())
        )[0]
            
        img = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(img)
        
        if pattern_type == 'circle':
            is_solid = True  # 默认实心，空心与数字0容易造成误判
            if is_solid:
                draw.ellipse([0, 0, size-1, size-1], fill=0)
            else:
                line_width = int(size * random.uniform(0.05, 0.15))
                draw.ellipse([0, 0, size-1, size-1], outline=0, width=line_width)
                
        elif pattern_type == 'hexagon':
            # 六边形噪声
            is_solid = random.random() < 0.7
            # is_solid = True  # 默认实心
            # 计算六边形的顶点
            center_x = size / 2
            center_y = size / 2
            radius = size * random.uniform(0.3, 0.45)  # 随机半径
            points = []
            for i in range(6):
                angle = i * 60  # 六边形每个角是60度
                x = center_x + radius * math.cos(math.radians(angle))
                y = center_y + radius * math.sin(math.radians(angle))
                points.append((x, y))
            
            if is_solid:
                draw.polygon(points, fill=0)
            else:
                line_width = int(size * random.uniform(0.05, 0.2))
                draw.line(points + [points[0]], fill=0, width=line_width)
                
        elif pattern_type == 'triangle':
            # 三角形噪声
            is_solid = True  # 默认实心
            # 生成三角形的三个顶点
            margin = size * 0.1  # 边距
            points = [
                (random.uniform(margin, size-margin), 
                 random.uniform(margin, size-margin)) 
                for _ in range(3)
            ]
            
            if is_solid:
                draw.polygon(points, fill=0)
            else:
                line_width = int(size * random.uniform(0.15, 0.25))
                draw.line(points + [points[0]], fill=0, width=line_width)
                
        elif pattern_type == 'vertical_stripe':
            num_stripes = random.randint(1, 1)
            for _ in range(num_stripes):
                stripe_width = int(size * random.uniform(0.1, 0.3))
                stripe_height = int(size * random.uniform(0.6, 1.0))
                x = random.randint(0, size - stripe_width)
                y = random.randint(0, size - stripe_height)
                draw.rectangle([x, y, x + stripe_width - 1, y + stripe_height - 1], fill=0)
                
        elif pattern_type == 'horizontal_stripe':
            num_stripes = random.randint(1, 1)
            for _ in range(num_stripes):
                stripe_width = int(size * random.uniform(0.6, 1.0))
                stripe_height = int(size * random.uniform(0.1, 0.3))
                x = random.randint(0, size - stripe_width)
                y = random.randint(0, size - stripe_height)
                draw.rectangle([x, y, x + stripe_width - 1, y + stripe_height - 1], fill=0)
                
        elif pattern_type == 'rectangle':
            is_solid = random.random() < 0.7
            rect_width = int(size * random.uniform(0.4, 0.8))
            rect_height = int(size * random.uniform(0.4, 0.8))
            x = random.randint(0, size - rect_width)
            y = random.randint(0, size - rect_height)
            if is_solid:
                draw.rectangle([x, y, x + rect_width - 1, y + rect_height - 1], fill=0)
            else:
                line_width = int(size * random.uniform(0.05, 0.3))
                draw.rectangle([x, y, x + rect_width - 1, y + rect_height - 1], 
                             outline=0, width=line_width)
        
        return np.array(img)

    def generate_image(self, input_dir: str) -> Tuple[np.ndarray, List[Tuple]]:
        """生成一张包含多个数字、字母和噪声图案的增强图像"""
        # 创建空白画布
        canvas = np.full((self.canvas_size, self.canvas_size), 255, dtype=np.uint8)
        
        # 随机决定数字和噪声数量
        num_digits = random.randint(self.min_digits, self.max_digits)
        num_noise_patterns = int(num_digits * 0.5)  # 噪声数量为数字数量的一半
        
        # 准备所有需要放置的项目
        placement_items = []  # [(type, img, digit, size), ...]
        
        # 获取所有可用的字符文件夹
        char_folders = os.listdir(input_dir)
        
        # 重置字母计数
        self.total_letters = 0
        
        # 准备数字和字母
        for _ in range(num_digits):
            # 随机选择一个字符文件夹
            folder = random.choice(char_folders)
            folder_path = os.path.join(input_dir, folder)
            
            if not os.path.isdir(folder_path):
                continue
                
            # 确定字符类型和标识
            if folder.isdigit():  # 数字文件夹
                char_type = 'digit'
                char_id = int(folder)
            elif (folder.startswith('upper_') or folder.startswith('lower_')):  # 字母
                if self.total_letters >= self.letter_count:  # 如果已经有2个字母，跳过
                    continue
                
                if folder.startswith('upper_'):
                    char_type = 'upper'
                    char_id = ord(folder[6:]) - ord('A') + 10
                else:  # lower_
                    char_type = 'lower'
                    char_id = ord(folder[6:]) - ord('a') + 36
                    
                self.total_letters += 1
            else:
                continue
            
            # 随机选择字符图像
            char_files = os.listdir(folder_path)
            char_file = random.choice(char_files)
            char_path = os.path.join(folder_path, char_file)
            
            # 加载和调整字符大小
            char_img = self._load_digit(char_path)  # 复用数字加载函数
            char_size = int(self.canvas_size * random.uniform(self.min_scale, self.max_scale))
            char_img = self._resize_digit(char_img, char_size)  # 复用数字缩放函数
            char_img = self._apply_augmentations(char_img)
            
            placement_items.append((char_type, char_img, char_id, char_size))
        
        # 准备噪声图案
        for _ in range(num_noise_patterns):
            noise_size = int(self.canvas_size * random.uniform(self.min_scale, self.max_scale*3))
            noise_img = self._generate_noise_pattern(noise_size)
            noise_img = self._apply_augmentations(noise_img)
            
            placement_items.append(('noise', noise_img, None, noise_size))
        
        # 随机打乱放置顺序
        random.shuffle(placement_items)
        
        # 记录已放置的位置和YOLO标注
        occupied_positions = []
        yolo_annotations = []
        
        # 放置所有项目
        for item_type, img, char_id, size in placement_items:
            try:
                x, y = self._find_valid_position(canvas, img, occupied_positions)
                # 只复制非白色像素
                mask = img < 255
                canvas[y:y+size, x:x+size][mask] = img[mask]
                occupied_positions.append((x, y, img))
                
                # 生成YOLO标注
                if item_type == 'digit' or (self.annotate_letters and item_type in ['upper', 'lower']):
                    center_x = (x + size/2) / self.canvas_size
                    center_y = (y + size/2) / self.canvas_size
                    width = size / self.canvas_size
                    height = size / self.canvas_size
                    
                    yolo_annotations.append((char_id, center_x, center_y, width, height))
                
            except ValueError:
                continue  # 如果找不到有效位置，跳过当前项目
        
        # 添加工业环境背景噪声
        canvas = self._add_industrial_background(canvas)
        
        return canvas, yolo_annotations

def generate_single_image(args):
    """生成单张图像的函数(用于多进程)"""
    i, input_dir, output_dir, augmentor = args
    image, annotations = augmentor.generate_image(input_dir)
    
    # 保存图像和标签
    image_path = os.path.join(output_dir, f"image_{i:06d}.png")
    label_path = os.path.join(output_dir, f"image_{i:06d}.txt")
    
    Image.fromarray(image).save(image_path)
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

def generate_dataset(
    input_dir: str,
    output_dir: str,
    num_images: int,
    augmentor: ImageAugmentor
):
    """生成数据集(多进程版本)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取CPU核心数
    num_cores = multiprocessing.cpu_count()
    # 创建进程池
    pool = Pool(processes=num_cores)
    
    # 准备参数
    args_list = [(i, input_dir, output_dir, augmentor) for i in range(num_images)]
    
    # 使用进度条
    from tqdm import tqdm
    print(f"使用 {num_cores} 个进程生成数据集...")
    
    # 使用进程池映射任务
    list(tqdm(
        pool.imap(generate_single_image, args_list),
        total=num_images,
        desc="生成数据集"
    ))
    
    # 关闭进程池
    pool.close()
    pool.join()

if __name__ == "__main__":
    # 配置参数
    config = {
        "canvas_size": 256,          # 输出图像的尺寸大小，生成 canvas_size x canvas_size 的正方形图像
        "background_noise_type": "perlin",  # 背景噪声类型：
                                           # - 'perlin': 柏林噪声，生成连续的、自然的纹理
                                           # - 'simplex': 单纯形噪声，类似柏林噪声但性能更好
                                           # - 'gaussian': 高斯噪声，完全随机的噪点
        "background_noise_intensity": 0.9,  # 背景噪声强度，范围 0.0-1.0
                                            # 值越大，背景噪声越明显
        "digit_noise_intensity_range": (0.0, 0.1),  # 数字噪声强度范围，范围 0.0-1.0
                                                    # 每个数字会随机选择这个范围内的噪声强度
        "min_digits": 5,                   # 每张图像最少数字数量
        "max_digits": 15,                  # 每张图像最多数字数量
        "min_scale": 0.05,                # 数字最小缩放比例（相对于 canvas_size）
                                          # 例如：0.04 表示数字最小为画布的 4%
        "max_scale": 0.15,                # 数字最大缩放比例（相对于 canvas_size）
                                          # 例如：0.15 表示数字最大为画布的 15%
        "min_spacing": 10,                  # 数字之间的最小间距（像素）
        "max_placement_attempts": 100,      # 寻找有效放置位置的最大尝试次数
                                          # 超过此次数认为无法放置更多数字
        "use_real_background": True,      # 是否使用真实背景图替代生成的噪声背景
        "real_background_dir": "./NEU-DET/IMAGES",  # 真实背景图片目录路径
        "augmentation_types": ['noise' ,'occlusion','rotation','aspect_ratio','rotation', 'brightness'],  # 启用的数据增强类型：
                                                                     # - 'noise': 添加噪声
                                                                     # - 'occlusion': 随机遮挡
                                                                     # - 'distortion': 扭曲变形
                                                                     # - 'aspect_ratio': 改变长宽比
                                                                     # - 'rotation': 旋转
                                                                     # - 'brightness': 亮度调节
        "noise_types": ['gaussian', 'salt_pepper', 'speckle'],  # 启用的噪声类型：
                                                    # - 'gaussian': 高斯噪声
                                                    # - 'salt_pepper': 椒盐噪声
                                                    # - 'speckle': 斑点噪声
                                                    # - 'poisson': 泊松噪声
        "occlusion_prob": 0.6,  # 应用遮挡增强的概率，范围 0.0-1.0
        "distortion_range": (0.9, 1.1),  # 扭曲变形的范围
                                        # - 小于1: 压缩
                                        # - 大于1: 拉伸
        "brightness_range": (1.1, 1.7),  # 亮度调节的范围
                                        # - 小于1: 变暗
                                        # - 大于1: 变亮
        "noise_patterns": ['circle', 'vertical_stripe', 'horizontal_stripe', 'rectangle', 'hexagon', 'triangle'],  # 启用的噪声图案类型
                                        # - 'circle': 圆形（实心/空心）
                                        # - 'vertical_stripe': 竖条纹
                                        # - 'horizontal_stripe': 横条纹
                                        # - 'rectangle': 矩形（实心/空心）
                                        # - 'hexagon': 六边形（实心/空心）
                                        # - 'triangle': 三角形（实心/空心）
        "noise_pattern_weights": {       # 各种噪声图案的生成权重
            'circle': 0.2,              # 圆形的生成概率
            'vertical_stripe': 0.2,     # 竖条纹的生成概率
            'horizontal_stripe': 0.2,   # 横条纹的生成概率
            'rectangle': 0.2,          # 矩形的生成概率
            'hexagon': 0.1,              # 六边形的生成概率
            'triangle': 0.1              # 三角形的生成概率
        },
        "annotate_letters": True,    # 是否为字母生成YOLO标注
        "letter_count": 2  # 单张图片字母出现总数  
    }
    
    augmentor = ImageAugmentor(**config)
    
    # 生成数据集
    generate_dataset(
        input_dir="font_numbers",
        output_dir="augmented_dataset",
        num_images=500,
        augmentor=augmentor
    )
