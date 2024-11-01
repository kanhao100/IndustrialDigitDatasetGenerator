# DigitAugmentor

DigitAugmentor 是一个用于生成和增强数字图像数据集的工具。它支持多种数据增强技术，包括噪声添加、遮挡、变形、旋转、亮度调整等，并支持使用真实背景图像。

## 特性

- 支持多种噪声类型：高斯噪声、椒盐噪声、斑点噪声、泊松噪声
- 多种数据增强方法：遮挡、变形、长宽比调整、旋转、亮度调整
- 支持使用真实背景图像
- 生成的标注符合YOLO格式，便于目标检测模型的训练

## 安装

确保已安装以下Python库：

- numpy
- Pillow
- matplotlib

可以通过以下命令安装：

```bash
pip install numpy pillow matplotlib
```

## 使用方法

1. 配置参数：在`font_png_augmentation.py`中设置所需的参数。
2. 生成数据集：运行`font_png_augmentation.py`生成增强后的图像和标注。
3. 可视化标注：使用`visualize_annotations.py`查看生成的图像及其标注。

## 示例

## 贡献

欢迎提交问题和请求功能。如果您想贡献代码，请提交Pull Request。

## 许可证

该项目使用MIT许可证。详情请参阅LICENSE文件。