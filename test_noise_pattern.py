import matplotlib.pyplot as plt

from font_png_augmentation import ImageAugmentor
from default_config import DEFAULT_CONFIG

def test_noise_patterns():
    # 配置参数
    config = DEFAULT_CONFIG.copy()

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
    config = DEFAULT_CONFIG.copy()

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