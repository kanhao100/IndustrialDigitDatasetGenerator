# IndustrialDigitDatasetGenerator

IndustrialDigitDatasetGenerator 是一个专门用于生成工业场景下数字图像数据集的工具。它能够从系统自带字体中提取数字(0-9)，并通过多种数据增强技术生成适用于工业环境的合成数据集，支持YOLO格式的目标检测标注。本工具支持多线程处理，依赖库简单，模块化设计，大量参数引出可供调节，易于使用。

## 关键词 Keywords
 **工业数字图像**、**工业仪表检测**、**仪表自动化**、**仪表盘读数识别**、**工业仪表读数识别**、**数据增强**、**目标检测**、**YOLO标注**、 **字体提取** 、**游标卡尺自动读数**

 **Industrial Digital Image**、**Industrial Instrument Detection**、**Instrument Automation**、**Dial Reading Recognition**、**Industrial Instrument Reading Recognition**、**Data Augmentation**、**Object Detection**、**YOLO Annotation**、**Font Extraction**、**Caliper Reading**


## 主要功能

### 字体提取
- 自动扫描并提取系统中的字体文件(默认支持Windows系统,其他系统修改字体目录即可)
- 智能过滤不适用的特殊字体
- 自动裁剪和对齐数字图像

### 数据增强
- **噪声增强**
  - 高斯噪声
  - 椒盐噪声
  - 斑点噪声
  - 泊松噪声

- **图像变换**
  - 随机旋转
  - 透视变形
  - 长宽比调整
  - 灰度调整
  - 随机遮挡

- **工业背景**
  - 柏林噪声生成
  - 真实工业图像背景(NEU-DET)，可自行添加
  - 随机条纹和污点

- **随机图案干扰增强**
  - 圆形图案（实心/空心）
  - 矩形图案（实心/空心）
  - 三角形图案（实心/空心）
  - 六边形图案（实心/空心）
  - 垂直条纹图案
  - 水平条纹图案
  - 支持自定义图案颜色
  - 支持调节图案大小和数量
  - 支持调节图案透明度
  - 随机位置和旋转角度

- **字母干扰增强**
  - 从字体文件提取字母
  - 进行随机字母干扰或者标注（用于对抗表盘单位）



### 标注生成
- 自动生成YOLO格式标注,包含边界框和类别信息

## 安装要求

### 依赖库

确保已安装以下Python库：

- numpy
- Pillow
- matplotlib
- tqdm

可以通过以下命令安装：

```bash
pip install numpy pillow matplotlib tqdm
```

## 使用方法

### 1. 字体提取
```bash
python font_extractor.py
```
从系统中提取数字字体并保存为PNG格式。

### 2. 数据集生成
```bash
python font_png_augmentation.py
```
使用提取的字体生成带有工业背景的数字图像数据集。

### 3. 可视化验证
```bash
python visualize_annotations.py
```
可视化生成的图像及其YOLO格式标注。

## 详细配置参数说明

### 基础配置
- `canvas_size`: 输出图像的尺寸大小（正方形，默认256×256）
- `min_digits`/`max_digits`: 每张图像包含的数字数量范围（默认5-15个）
- `min_scale`/`max_scale`: 数字大小范围（相对于画布尺寸的比例，默认0.05-0.15）
- `min_spacing`: 数字间最小间距（像素，默认10）
- `max_placement_attempts`: 数字放置最大尝试次数（默认100）

### 背景配置
- `background_noise_type`: 背景噪声类型
  - `perlin`: 柏林噪声，生成连续的、自然的纹理
  - `simplex`: 单纯形噪声，类似柏林噪声但性能更好
  - `gaussian`: 高斯噪声，完全随机的噪点
- `background_noise_intensity`: 背景噪声强度（0.0-1.0，默认0.9）
- `use_real_background`: 是否使用真实背景图像
- `real_background_dir`: 真实背景图片目录路径

### 数据增强配置
- `augmentation_types`: 可选的数据增强类型
  - `noise`: 添加噪声
  - `occlusion`: 随机遮挡
  - `distortion`: 扭曲变形
  - `aspect_ratio`: 改变长宽比
  - `rotation`: 旋转
  - `brightness`: 亮度调节

- `noise_types`: 支持的噪声类型
  - `gaussian`: 高斯噪声
  - `salt_pepper`: 椒盐噪声
  - `speckle`: 斑点噪声
  - `poisson`: 泊松噪声

- `digit_noise_intensity_range`: 数字噪声强度范围（0.0-1.0）
- `occlusion_prob`: 遮挡概率（0.0-1.0，默认0.6）
- `distortion_range`: 扭曲变形范围（默认0.9-1.1）
- `brightness_range`: 亮度调节范围（默认1.1-1.7）

### 图案干扰配置
- `noise_patterns`: 支持的图案类型
  - `circle`: 圆形（实心/空心）
  - `vertical_stripe`: 竖条纹
  - `horizontal_stripe`: 横条纹
  - `rectangle`: 矩形（实心/空心）
  - `hexagon`: 六边形（实心/空心）
  - `triangle`: 三角形（实心/空心）

- `noise_pattern_weights`: 各图案生成权重
  - `circle`: 0.2
  - `vertical_stripe`: 0.2
  - `horizontal_stripe`: 0.2
  - `rectangle`: 0.2
  - `hexagon`: 0.1
  - `triangle`: 0.1

### 字母干扰配置
- `annotate_letters`: 是否为字母生成YOLO标注（默认True）
- `letter_count`: 单张图片字母出现总数（默认2）

## 项目结构
```
IndustrialDigitDatasetGenerator/
|—— NEU-DET                # 真实工业背景图片目录
|———— IMAGES               # 图片目录
|—— font_numbers           # 提取的字体图片目录
|———— 0
|———— 1
|———— 2
|———— 3
|———— 4
|———— 5
|———— 6
|———— 7
|———— 8
|———— 9
|—— augmented_dataset      # 生成的数据集目录
|—— font_extractor.py      # 字体提取工具
|—— font_png_augmentation.py # 数据集生成主程序
|—— visualize_annotations.py # 标注可视化工具
|—— requirements.txt         # 参考依赖库列表
└── README.md
```

## 贡献指南

欢迎提交 Issue 和 Pull Request。


## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 效果展示

### 数据增强效果
![数据增强效果](docs/images/Digital_Enhanced_sample.jpg)
![数据增强效果](docs/images/single_digit_augmentations_8.JPG)


### 图案干扰效果
![图案干扰效果](docs/images/test_noise_patterns.JPG)
*不同类型的随机图案干扰示例*

### YOLO标注可视化
![标注可视化](docs/images/visualize_yolo_annotations.JPG)
*YOLO格式标注框可视化效果*