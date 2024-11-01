import os
import numpy as np
from PIL import Image
import random
from typing import Tuple, List

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
                                          # 会被 placement_density 进一步调整
        placement_density: float = 0.7,        # 数字放置密度，范围 0.0-1.0
                                          # - 0.0: 最稀疏，数字间距最大
                                          # - 1.0: 最密集，数字间距最小
        max_placement_attempts: int = 100,      # 寻找有效放置位置的最大尝试次数
                                          # 超过此次数认为无法放置更多数字
        use_real_background: bool = False,      # 新增：是否使用真实背景图
        real_background_dir: str = "./NEU-DET/IMAGES",  # 新增：真实背景图目录
        augmentation_types: List[str] = None,  # 新增：指定要使用的增强类型
        noise_types: List[str] = None,  # 新增：指定要使用的噪声类型
        occlusion_prob: float = 0.3,  # 新增：遮挡概率
        distortion_range: Tuple[float, float] = (0.8, 1.2),  # 新增：变形范围
        brightness_range: Tuple[float, float] = (0.7, 1.3),  # 新增：亮度调节范围
    ):
        self.canvas_size = canvas_size
        self.background_noise_type = background_noise_type
        self.background_noise_intensity = background_noise_intensity
        self.digit_noise_intensity_range = digit_noise_intensity_range
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_spacing = int(min_spacing * (1 - placement_density))  # 根据密度调整间距
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
        if use_real_background:
            # 获取所有非patches开头的jpg文件
            self.real_background_files = [
                f for f in os.listdir(real_background_dir) 
                if f.endswith('.jpg') and not f.startswith('patches')
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
                    # 随机添加遮挡
                    w, h = img.size
                    occlude_w = int(w * random.uniform(0.1, 0.3))
                    occlude_h = int(h * random.uniform(0.1, 0.3))
                    x = random.randint(0, w - occlude_w)
                    y = random.randint(0, h - occlude_h)
                    # 创建白色矩形
                    occlude = Image.new('L', (occlude_w, occlude_h), 255)
                    img.paste(occlude, (x, y))
            
            elif aug_type == 'distortion':
                # 使用PIL的变形方法
                w, h = img.size
                # 计算扭曲变换的控制点
                distort_x = random.uniform(*self.distortion_range)
                distort_y = random.uniform(*self.distortion_range)
                
                # 定义8个系数的透视变换
                # 源点和目标点的对应关系
                width = w - 1
                height = h - 1
                
                coeffs = (
                    0, 0,                     # 左上角
                    width * distort_x, 0,     # 右上角
                    width * distort_x, height * distort_y,  # 右下角
                    0, height * distort_y     # 左下角
                )
                
                img = img.transform(
                    (w, h),
                    Image.QUAD,  # 使用四边形变换而不是透视变换
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
                # 将图像转换为numpy数组进行处理
                img_array = np.array(img)
                # 只对非白色区域调整亮度
                mask = img_array < 50  # 修改：使用更严格的阈值来识别数字区域
                if mask.any():  # 确保有非白色像素
                    brightness_factor = random.uniform(*self.brightness_range)
                    # 对非白色区域应用亮度调整
                    adjusted = img_array.copy()  # 修改：创建副本避免直接修改原数组
                    adjusted[mask] = np.clip(
                        img_array[mask] * brightness_factor,
                        0, 
                        50  # 修改：限制最大值，保持数字清晰度
                    ).astype(np.uint8)
                    img = Image.fromarray(adjusted)
        
        # 最后将PIL图像转回numpy数组
        return np.array(img)

    def _load_and_resize_digit(self, digit_path: str, target_size: int) -> np.ndarray:
        """加载并调整数字图片大小"""
        # 加载图像
        img = Image.open(digit_path).convert('L')
        img_array = np.array(img)
        
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

    def generate_image(self, input_dir: str) -> Tuple[np.ndarray, List[Tuple]]:
        """生成一张包含多个数字的增强图像，返回图像和YOLO格式的标注"""
        # 创建空白画布
        canvas = np.full((self.canvas_size, self.canvas_size), 255, dtype=np.uint8)
        
        # 随机决定数字数量
        num_digits = random.randint(self.min_digits, self.max_digits)
        
        # 记录已放置的数字位置和图像
        occupied_positions = []
        yolo_annotations = []  # 修改：使用YOLO格式的标注
        
        # 放置数字
        for _ in range(num_digits):
            digit = random.randint(0, 9)  # class_id 就是数字本身
            digit_dir = os.path.join(input_dir, str(digit))
            digit_files = os.listdir(digit_dir)
            digit_file = random.choice(digit_files)
            digit_path = os.path.join(digit_dir, digit_file)
            
            digit_size = int(self.canvas_size * random.uniform(self.min_scale, self.max_scale))
            digit_img = self._load_and_resize_digit(digit_path, digit_size)
            
            # 应用数据增强
            digit_img = self._apply_augmentations(digit_img)
            
            try:
                x, y = self._find_valid_position(canvas, digit_img, occupied_positions)
                # 只复制非白色像素
                mask = digit_img < 255
                canvas[y:y+digit_size, x:x+digit_size][mask] = digit_img[mask]
                occupied_positions.append((x, y, digit_img))
                
                # 计算YOLO格式的标注
                # 中心点坐标
                center_x = (x + digit_size/2) / self.canvas_size
                center_y = (y + digit_size/2) / self.canvas_size
                # 宽高
                width = digit_size / self.canvas_size
                height = digit_size / self.canvas_size
                
                # 添加YOLO格式标注：class_id center_x center_y width height
                yolo_annotations.append((digit, center_x, center_y, width, height))
                
            except ValueError:
                break
        
        # 添加工业环境背景噪声
        canvas = self._add_industrial_background(canvas)
        
        return canvas, yolo_annotations

def generate_dataset(
    input_dir: str,
    output_dir: str,
    num_images: int,
    augmentor: ImageAugmentor
):
    """生成数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        image, annotations = augmentor.generate_image(input_dir)
        # 保存图像和标签
        image_path = os.path.join(output_dir, f"image_{i:06d}.png")
        label_path = os.path.join(output_dir, f"image_{i:06d}.txt")
        
        Image.fromarray(image).save(image_path)
        # 保存YOLO格式的标注
        with open(label_path, 'w') as f:
            for ann in annotations:
                # 每行格式：class_id center_x center_y width height
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

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
        "min_spacing": 5,                  # 数字之间的最小间距（像素）
                                          # 会被 placement_density 进一步调整
        "placement_density": 0.8,        # 数字放置密度，范围 0.0-1.0
                                          # - 0.0: 最稀疏，数字间距最大
                                          # - 1.0: 最密集，数字间距最小
        "max_placement_attempts": 100,      # 寻找有效放置位置的最大尝试次数
                                          # 超过此次数认为无法放置更多数字
        "use_real_background": True,      # 是否使用真实背景图替代生成的噪声背景
        "real_background_dir": "./NEU-DET/IMAGES",  # 真实背景图片目录路径
        "augmentation_types": ['noise' ,'occlusion','rotation','aspect_ratio','brightness'],  # 启用的数据增强类型：
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
        "occlusion_prob": 0.5,  # 应用遮挡增强的概率，范围 0.0-1.0
        "distortion_range": (0.9, 1.1),  # 扭曲变形的范围
                                        # - 小于1: 压缩
                                        # - 大于1: 拉伸
        "brightness_range": (0.3, 0.6),  # 亮度调节的范围
                                        # - 小于1: 变暗
                                        # - 大于1: 变亮
    }
    
    augmentor = ImageAugmentor(**config)
    
    # 生成数据集
    generate_dataset(
        input_dir="font_numbers",
        output_dir="augmented_dataset",
        num_images=200,
        augmentor=augmentor
    )
