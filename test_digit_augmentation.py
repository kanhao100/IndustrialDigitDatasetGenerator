from font_png_augmentation import ImageAugmentor
import matplotlib.pyplot as plt
import os
import random

def test_digit_augmentation():
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
        "noise_patterns": ['circle', 'vertical_stripe', 'horizontal_stripe', 'rectangle'],  # 启用的噪声图案类型
                                        # - 'circle': 圆形（实心/空心）
                                        # - 'vertical_stripe': 竖条纹
                                        # - 'horizontal_stripe': 横条纹
                                        # - 'rectangle': 矩形（实心/空心）
        "noise_pattern_weights": {       # 各种噪声图案的生成权重
            'circle': 0.25,              # 圆形的生成概率
            'vertical_stripe': 0.25,     # 竖条纹的生成概率
            'horizontal_stripe': 0.25,   # 横条纹的生成概率
            'rectangle': 0.25            # 矩形的生成概率
        },
    }
    augmentor = ImageAugmentor(**config)
    
    # 为每个数字生成多个增强样本
    digits = range(10)  # 0-9
    samples_per_digit = 10
    rows = len(digits)
    cols = samples_per_digit
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 18))
    fig.suptitle('Digital enhanced sample', fontsize=16)
    
    input_dir = "font_numbers"  # 字体图像目录
    
    for row, digit in enumerate(digits):
        digit_dir = os.path.join(input_dir, str(digit))
        digit_files = os.listdir(digit_dir)
        
        for col in range(cols):
            # 随机选择一个字体文件
            digit_file = random.choice(digit_files)
            digit_path = os.path.join(digit_dir, digit_file)
            
            # 使用新的两步加载和调整大小的逻辑
            digit_img = augmentor._load_digit(digit_path)
            digit_size = 64  # 固定大小以便比较
            digit_img = augmentor._resize_digit(digit_img, digit_size)
            
            # 应用数据增强
            digit_img = augmentor._apply_augmentations(digit_img)
            
            # 显示图像
            ax = axes[row, col]
            ax.imshow(digit_img, cmap='gray')
            if col == 0:  # 只在每行第一个图案处显示数字
                ax.set_title(f'Digit {digit}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_single_digit_augmentations():
    """测试单个数字的不同增强效果"""
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
        "noise_patterns": ['circle', 'vertical_stripe', 'horizontal_stripe', 'rectangle'],  # 启用的噪声图案类型
                                        # - 'circle': 圆形（实心/空心）
                                        # - 'vertical_stripe': 竖条纹
                                        # - 'horizontal_stripe': 横条纹
                                        # - 'rectangle': 矩形（实心/空心）
        "noise_pattern_weights": {       # 各种噪声图案的生成权重
            'circle': 0.25,              # 圆形的生成概率
            'vertical_stripe': 0.25,     # 竖条纹的生成概率
            'horizontal_stripe': 0.25,   # 横条纹的生成概率
            'rectangle': 0.25            # 矩形的生成概率
        },
    }
    augmentor = ImageAugmentor(**config)
    
    # 选择一个数字进行详细测试
    test_digit = 8 
    input_dir = "font_numbers"
    digit_dir = os.path.join(input_dir, str(test_digit))
    digit_files = os.listdir(digit_dir)
    digit_file = digit_files[0]  # 使用第一个字体文件
    digit_path = os.path.join(digit_dir, digit_file)
    
    # 显示4x4的增强效果
    rows, cols = 6, 6
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(f'Various enhanced effects for the digit {test_digit}', fontsize=16)
    
    digit_size = 64
    for i in range(rows):
        for j in range(cols):
            # 使用新的两步加载和调整大小的逻辑
            digit_img = augmentor._load_digit(digit_path)
            digit_img = augmentor._resize_digit(digit_img, digit_size)
            
            # 应用增强
            digit_img = augmentor._apply_augmentations(digit_img)
            
            ax = axes[i, j]
            ax.imshow(digit_img, cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试所有数字的增强效果
    # test_digit_augmentation()
    
    # 测试单个数字的多种增强效果
    test_single_digit_augmentations() 