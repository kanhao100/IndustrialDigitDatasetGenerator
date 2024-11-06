# ImageAugmentor类的默认参数配置，请修改 | Default parameters for ImageAugmentor class, please modify
DEFAULT_CONFIG = {
        "real_background_dir": "./NEU-DET/IMAGES",  # 真实背景图片目录路径 | Path to real background images directory
        "canvas_size": 256,          # 输出图像的尺寸大小，生成 canvas_size x canvas_size 的正方形图像 | Output image size (square)
        "background_noise_type": "perlin",  # 背景噪声类型 | Background noise type: perlin/simplex/gaussian
                                            # - 'perlin': 柏林噪声，生成连续的、自然的纹理
                                            # - 'simplex': 单纯形噪声，类似柏林噪声但性能更好
                                            # - 'gaussian': 高斯噪声，完全随机的噪点
        "background_noise_intensity": 0.9,  # 背景噪声强度，范围 0.0-1.0 | Background noise intensity, range 0.0-1.0
        "digit_noise_intensity_range": (0.0, 0.1),  # 数字噪声强度范围，范围 0.0-1.0 | Digit noise intensity range
        "min_digits": 10,                   # 每张图像最少数字数量 | Minimum number of digits per image
        "max_digits": 20,                  # 每张图像最多数字数量 | Maximum number of digits per image
        "min_scale": 0.04,                # 数字最小缩放比例 | Minimum digit scale relative to canvas
        "max_scale": 0.15,                # 数字最大缩放比例 | Maximum digit scale relative to canvas
        "min_spacing": 10,                  # 数字之间的最小间距（像素）| Minimum spacing between digits (pixels)
        "max_placement_attempts": 100,      # 寻找有效放置位置的最大尝试次数 | Maximum attempts to place a digit
        "use_real_background": True,      # 是否使用真实背景图替代生成的噪声背景 | Whether to use real background images
        "augmentation_types": ['noise' ,'occlusion','rotation','aspect_ratio','rotation', 'brightness'],  # 启用的数据增强类型 | Enabled augmentation types
                                                                            # - 'noise': 添加噪声
                                                                            # - 'occlusion': 随机遮挡
                                                                            # - 'distortion': 扭曲变形
                                                                            # - 'aspect_ratio': 改变长宽比
                                                                            # - 'rotation': 旋转
                                                                            # - 'brightness': 亮度调节
        "noise_types": ['gaussian', 'salt_pepper', 'speckle'],  # 启用的噪声类型 | Enabled noise types
                                                            # - 'gaussian': 高斯噪声
                                                            # - 'salt_pepper': 椒盐噪声
                                                            # - 'speckle': 斑点噪声
                                                            # - 'poisson': 泊松噪声
        "occlusion_prob": 0.7,  # 应用遮挡增强的概率 | Probability of applying occlusion augmentation
        "distortion_range": (0.9, 1.1),  # 扭曲变形的范围 (小于1:压缩 大于1:拉伸) | Distortion range (<1: compress, >1: stretch)
        "brightness_range": (1.1, 1.7),  # 亮度调节的范围 (小于1:变暗 大于1:变亮) | Brightness adjustment range (<1: darker, >1: brighter)
        "noise_patterns": ['circle', 'vertical_stripe', 'horizontal_stripe', 'rectangle', 'hexagon', 'triangle'],  # 启用的噪声图案类型 | Enabled noise pattern types
                                            # - 'circle': 圆形（实心/空心）
                                            # - 'vertical_stripe': 竖条纹
                                            # - 'horizontal_stripe': 横条纹
                                            # - 'rectangle': 矩形（实心/空心）
                                            # - 'hexagon': 六边形（实心/空心）
                                            # - 'triangle': 三角形（实心/空心）
        "noise_pattern_weights": {       # 各种噪声图案的生成权重 | Generation weights for noise patterns
            'circle': 0.2,              # 圆形的生成概率 | Probability for circle pattern
            'vertical_stripe': 0.1,     # 竖条纹的生成概率 | Probability for vertical stripe pattern
            'horizontal_stripe': 0.2,   # 横条纹的生成概率 | Probability for horizontal stripe pattern
            'rectangle': 0.2,          # 矩形的生成概率 | Probability for rectangle pattern
            'hexagon': 0.1,              # 六边形的生成概率 | Probability for hexagon pattern
            'triangle': 0.2              # 三角形的生成概率 | Probability for triangle pattern
        },
        "char_weights": {               # 字符权重 | Character weights
            '0': 1.0,
            '1': 3.0,  # 数字1的识别难度较大，建议增加其权重 | Increased weight for digit 1 due to recognition difficulty
            '2': 1.0,
            '3': 1.0,
            '4': 1.0,
            '5': 1.0,
            '6': 1.0,
            '7': 1.5,
            '8': 1.0,
            '9': 1.0,
            'upper': 1.0,  # 大写字母出现概率 | Probability for uppercase letters
            'lower': 1.0,  # 小写字母出现概率 | Probability for lowercase letters
        },
        "annotate_letters": True,    # 是否为字母生成YOLO标注 | Whether to generate YOLO annotations for letters
        "letter_count": 2,  # 单张图片字母出现总数 | Total number of letters per image
        "augmentation_prob": 0.9,  # 应用增强的概率 | Probability of applying augmentation
        "seed": None,  # 单线程的随机种子参数 | Random seed for single-threaded execution
    }




# 字体提取参数配置 | Font extraction parameter configuration
IMAGE_SIZE = (64, 64)  # 输出图像大小 | Output image size
TEXT_COLOR = 'black'   # 文字颜色 | Text color
OUTPUT_BASE_DIR = "font_numbers"  # 输出目录名 | Output directory name
DEFAULT_CHARS = [str(i) for i in range(10)] + list("cmCM")  # 要提取的字符，默认0-9和cmCM | Characters to extract, default 0-9 and cmCM


def is_wsl():
    """检测是否在 WSL 环境中运行 | Check if running in WSL environment"""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False
# 根据环境确定字体目录 | Determine font directory based on environment
if is_wsl():
    # WSL 环境下的字体目录 | Font directory for WSL environment
    FONT_DIRECTORIES = [
        '/mnt/c/Windows/Fonts',
        '/mnt/c/Windows/Boot/Fonts',
        '/mnt/c/Program Files/Microsoft Office/root/vfs/Fonts',
        '/mnt/c/Program Files (x86)/Microsoft Office/root/vfs/Fonts',
        '/mnt/c/Program Files/Common Files/Microsoft Shared/Fonts',
        '/mnt/c/Program Files (x86)/Common Files/Microsoft Shared/Fonts',
        '/mnt/c/Program Files (x86)/Microsoft Office/root/vfs/Fonts/private',
        '/mnt/d/number_pic_dataset_ours/font_files', # 自定义的字体文件目录 | Custom font file directory
    ]
else:
    # Windows 环境下的字体目录 | Font directory for Windows environment
    FONT_DIRECTORIES = [
        "C:\\Windows\\Fonts",
        "C:\\Windows\\Boot\\Fonts",
        "C:\\Program Files\\Microsoft Office\\root\\vfs\\Fonts",
        "C:\\Program Files (x86)\\Microsoft Office\\root\\vfs\\Fonts",
        "C:\\Program Files\\Common Files\\Microsoft Shared\\Fonts",
        "C:\\Program Files (x86)\\Common Files\\Microsoft Shared\\Fonts",
        "C:\\Program Files (x86)\\Microsoft Office\\root\\vfs\\Fonts\\private",
        "D:\\anaconda3\\Lib\\site-packages\\navigator_updater\\static\\fonts",
        "D:\\anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf",
        "D:\\JetBrains\\PyCharm Community Edition 2023.1.1\\jbr\\lib\\fonts",
        "C:\\Program Files\\Azure Data Studio\\resources\\app\\extensions\\markdown-math\\notebook-out\\fonts",
        "D:\\number_pic_dataset_ours\\font_files", # 自定义的字体文件目录 | Custom font file directory
                                                   # 请将自行下载字体文件放在该目录下 | Please place the downloaded font files in this directory
    ]

# 需要排除的字体列表（不区分大小写） | Font list to exclude (case-insensitive)
EXCLUDED_FONTS = {
    'bssym7', 'holomdl2', 'marlett', 'inkfree', 'javatext',
    'mtextra', 'refspcl', 'segmdl2', 'segoepr', 'segoeprb',
    'segoesc', 'segoescb', 'stcaiyun', 'sthupo', 'symbol',
    'webdings', 'wingding', 'wingdng2', 'BRADHITC', 'ITCKRIST',
    'MISTRAL', 'mvboli', 'PAPYRUS', 'PRISTINA', 'FREESCPT',
    'cmex10', 'cmsy10', 'DejaVuSansDisplay', 'DejaVuSerifDisplay',
    'KaTeX_AMS-Regular', 'KaTeX_Caligraphic-Bold', 'KaTeX_Caligraphic-Regular',
    'KaTeX_Script-Regular', 'KaTeX_Size1-Regular', 'KaTeX_Size2-Regular',
    'KaTeX_Size3-Regular', 'KaTeX_Size4-Regular', 'STIXNonUniBolIta',
    'STIXNonUniBol', 'STIXNonUniIta', 'STIXNonUni', 'STIXSizFiveSymReg',
    'STIXSizFourSymReg', 'STIXSizOneSymReg', 'STIXSizThreeSymReg',
    'STIXSizTwoSymReg', 'chs_boot', 'cht_boot', 'jpn_boot', 'kor_boot', 'wgl4_boot',
    'Strebd3'
}

# 需要排除的字体的关键字（部分匹配，不区分大小写） | Font keywords to exclude (partial match, case-insensitive)
EXCLUDED_KEYWORDS = {'symbol', 'wing', 'webding'}

# Windows常用字体列表（包括变体） | Common Windows fonts (including variants)
WINDOWS_FONTS = {
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
    
    # 中文字体及其变体 | Chinese fonts and variants
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



