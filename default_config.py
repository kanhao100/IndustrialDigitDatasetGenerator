# 默认配置参数, 请修改
DEFAULT_CONFIG = {
        "canvas_size": 256,          # 输出图像的尺寸大小，生成 canvas_size x canvas_size 的正方形图像
        "background_noise_type": "perlin",  # 背景噪声类型：
                                           # - 'perlin': 柏林噪声，生成连续的、自然的纹理
                                           # - 'simplex': 单纯形噪声，类似柏林噪声但性能更好
                                           # - 'gaussian': 高斯噪声，完全随机的噪点
        "background_noise_intensity": 0.9,  # 背景噪声强度，范围 0.0-1.0
                                            # 值越大，背景噪声越明显
        "digit_noise_intensity_range": (0.0, 0.1),  # 数字噪声强度范围，范围 0.0-1.0
                                                    # 每个数字会随机选择这个范围内的噪声强度
        "min_digits": 10,                   # 每张图像最少数字数量
        "max_digits": 20,                  # 每张图像最多数字数量
        "min_scale": 0.03,                # 数字最小缩放比例（相对于 canvas_size）
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
        "occlusion_prob": 0.7,  # 应用遮挡增强的概率，范围 0.0-1.0
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
            'vertical_stripe': 0.1,     # 竖条纹的生成概率
            'horizontal_stripe': 0.2,   # 横条纹的生成概率
            'rectangle': 0.2,          # 矩形的生成概率
            'hexagon': 0.1,              # 六边形的生成概率
            'triangle': 0.2              # 三角形的生成概率
        },
        "annotate_letters": True,    # 是否为字母生成YOLO标注
        "letter_count": 2  # 单张图片字母出现总数  
    }




# 字体提取参数配置
IMAGE_SIZE = (64, 64)  # 输出图像大小
TEXT_COLOR = 'black'   # 文字颜色
OUTPUT_BASE_DIR = "font_numbers"  # 输出目录名
DEFAULT_CHARS = [str(i) for i in range(10)] + list("cmCM")  # 要提取的字符，默认0-9和cmCM


def is_wsl():
    """检测是否在 WSL 环境中运行"""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False
# 根据环境确定字体目录
if is_wsl():
    # WSL 环境下的字体目录
    FONT_DIRECTORIES = [
        '/mnt/c/Windows/Fonts',
        '/mnt/c/Program Files/Microsoft Office/root/vfs/Fonts',
        '/mnt/c/Program Files (x86)/Microsoft Office/root/vfs/Fonts',
        '/mnt/c/Program Files/Common Files/Microsoft Shared/Fonts',
        '/mnt/c/Program Files (x86)/Common Files/Microsoft Shared/Fonts',
    ]
else:
    # Windows 环境下的字体目录
    FONT_DIRECTORIES = [
        "C:\\Windows\\Fonts",
        "C:\\Program Files\\Microsoft Office\\root\\vfs\\Fonts",
        "C:\\Program Files (x86)\\Microsoft Office\\root\\vfs\\Fonts",
        "C:\\Program Files\\Common Files\\Microsoft Shared\\Fonts",
        "C:\\Program Files (x86)\\Common Files\\Microsoft Shared\\Fonts",
    ]

# 需要排除的字体列表（不区分大小写）
EXCLUDED_FONTS = {
    'bssym7', 'holomdl2', 'marlett', 'inkfree', 'javatext',
    'mtextra', 'refspcl', 'segmdl2', 'segoepr', 'segoeprb',
    'segoesc', 'segoescb', 'stcaiyun', 'sthupo', 'symbol',
    'webdings', 'wingding', 'wingdng2', 'BRADHITC', 'ITCKRIST',
    'MISTRAL', 'mvboli', 'PAPYRUS', 'PRISTINA', 'FREESCPT'
}

# 需要排除的字体的关键字（部分匹配，不区分大小写）
EXCLUDED_KEYWORDS = {'symbol', 'wing', 'webding'}

# Windows常用字体列表（包括变体）
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



