# IndustrialDigitDatasetGenerator

IndustrialDigitDatasetGenerator 是一个专门用于生成工业场景下数字图像数据集的工具。它能够从系统字体中提取数字，并通过多种数据增强技术生成适用于工业环境的合成数据集，支持YOLO格式的目标检测标注。

## 主要功能

### 字体处理
- 自动扫描并提取系统中的字体文件
- 智能过滤不适用的特殊字体
- 支持多种字体格式（TTF、OTF、TTC）
- 自动裁剪和对齐数字图像

### 数据增强
- **噪声增强**
  - 高斯噪声
  - 椒盐噪声
  - 斑点噪声
  - 泊松噪声

- **图像变换**
  - 随机旋转
  - 透视变换
  - 长宽比调整
  - 亮度调整
  - 随机遮挡

- **工业背景**
  - 柏林噪声生成
  - 真实工业图像背景
  - 随机条纹和污点

### 标注生成
- 自动生成YOLO格式标注
- 支持多目标检测
- 包含边界框和类别信息

## 安装要求

### 依赖库

确保已安装以下Python库：

- numpy
- Pillow
- matplotlib

可以通过以下命令安装：

```bash
pip install numpy pillow matplotlib
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

## 配置说明

主要参数可在 `font_png_augmentation.py` 中配置：

- `canvas_size`: 输出图像尺寸
- `min_digits`/`max_digits`: 每张图像的数字数量范围
- `min_scale`/`max_scale`: 数字大小范围
- `background_noise_type`: 背景噪声类型
- `use_real_background`: 是否使用真实背景图像

## 项目结构
```
IndustrialDigitDatasetGenerator/
├── font_extractor.py        # 字体提取工具
├── font_png_augmentation.py # 数据集生成主程序
├── visualize_annotations.py # 标注可视化工具
└── README.md
```

## 贡献指南

欢迎提交 Issue 和 Pull Request。在提交代码前，请确保：
1. 代码风格符合 PEP 8 规范
2. 添加必要的注释和文档
3. 所有测试用例通过

## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。