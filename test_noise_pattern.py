from font_png_augmentation import ImageAugmentor
import matplotlib.pyplot as plt

def test_noise_patterns():
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
    }
    augmentor = ImageAugmentor(**config)
    
    # 为每种类型生成多个样本
    patterns = ['circle', 'vertical_stripe', 'horizontal_stripe', 'rectangle', 'hexagon', 'triangle']
    samples_per_pattern = 6
    rows = len(patterns)
    cols = samples_per_pattern
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    fig.suptitle('Noise pattern sample', fontsize=16)
    
    for row, pattern in enumerate(patterns):
        # 临时修改权重以生成指定类型的图案
        temp_weights = {p: 1.0 if p == pattern else 0.0 for p in patterns}
        augmentor.noise_pattern_weights = temp_weights
        
        for col in range(cols):
            # 生成噪声图案
            noise_img = augmentor._generate_noise_pattern(size=64)
            # 应用数据增强
            noise_img = augmentor._apply_augmentations(noise_img)
            
            # 显示图案
            ax = axes[row, col]
            ax.imshow(noise_img, cmap='gray')
            if col == 0:  # 只在每行第一个图案处显示类型名称
                ax.set_title(f'{pattern}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 恢复原始权重
    augmentor.noise_pattern_weights = config['noise_pattern_weights']

def test_mixed_patterns():
    """测试随机混合的噪声图案"""
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
                                          # 会被 placement_density 进一步调整
        "placement_density": 0.8,        # 数字放置密度，范围 0.0-1.0
                                          # - 0.0: 最稀疏，数字间距最大
                                          # - 1.0: 最密集，数字间距最小
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
    }    
    augmentor = ImageAugmentor(**config)
    
    # 显示4x4的随机图案
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle('Randomly mixed noise patterns', fontsize=16)
    
    for i in range(rows):
        for j in range(cols):
            noise_img = augmentor._generate_noise_pattern(size=64)
            noise_img = augmentor._apply_augmentations(noise_img)
            
            ax = axes[i, j]
            ax.imshow(noise_img, cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试每种类型的多个样本
    test_noise_patterns()
    
    # 测试随机混合的图案
    # test_mixed_patterns() 