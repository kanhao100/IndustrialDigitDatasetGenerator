import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
from torchvision.utils import make_grid
import cv2

class ModelEvaluator:
    def __init__(self, model, device, class_names, test_dir, output_dir):
        """
        初始化模型评估器
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.test_dir = test_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 数据预处理
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        
        # 存储预测结果
        self.predictions = []
        self.true_labels = []
        self.failed_examples = {cls: [] for cls in class_names}
        self.success_examples = {cls: [] for cls in class_names}

    def evaluate_model(self):
        """评估模型性能"""
        self.model.eval()
        
        with torch.no_grad():
            for class_name in os.listdir(self.test_dir):
                class_dir = os.path.join(self.test_dir, class_name)
                
                # 检查是否为目录
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        # print(f"正在处理: {img_name}")
                        
                        # 检查是否为文件
                        if os.path.isfile(img_path):
                            true_label = class_name  # 文件夹名即为真实标签
                            # print(f"正在处理: {img_name}, 真实标签: {true_label}")
                            
                            # 预测
                            image = Image.open(img_path).convert('RGB')
                            input_tensor = self.data_transforms(image).unsqueeze(0).to(self.device)
                            outputs = self.model(input_tensor)
                            _, preds = torch.max(outputs, 1)
                            predicted_class = self.class_names[preds.item()]
                            
                            # 保存预测结果
                            self.predictions.append(predicted_class)
                            self.true_labels.append(true_label)
                            
                            # 保存预测成功和失败的例子
                            if predicted_class == true_label:
                                self.success_examples[true_label].append(img_path)
                            else:
                                self.failed_examples[true_label].append((img_path, predicted_class))
                            
                            # 复制并重命名图片
                            # new_filename = f"{predicted_class}_{img_name}"
                            # output_path = os.path.join(self.output_dir, new_filename)
                            # shutil.copy(img_path, output_path)

    def plot_confusion_matrix(self):
        """绘制并保存混淆矩阵"""
        cm = confusion_matrix(self.true_labels, self.predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

    def print_classification_report(self):
        """打印分类报告"""
        report = classification_report(self.true_labels, self.predictions)
        print("\n分类报告:")
        print(report)
        
        # 保存报告到文件
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

    def create_example_grid(self, examples_dict, is_success=True, samples_per_class=5):
        """创建示例图片网格"""
        all_images = []
        
        for class_name in self.class_names:
            examples = examples_dict[class_name]
            if not examples:
                continue
                
            # 随机选择样本
            selected = random.sample(examples, min(samples_per_class, len(examples)))
            
            for example in selected:
                if is_success:
                    img_path = example
                    label = f"True: {class_name}"
                else:
                    img_path, pred = example
                    label = f"True: {class_name}, Pred: {pred}"
                
                # 读取和调整图像
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                # 添加标签
                img = cv2.putText(img, label, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                all_images.append(img)
        
        if not all_images:
            return None
            
        # 创建网格
        rows = len(self.class_names)
        cols = samples_per_class
        grid_size = (rows, cols)
        
        # 填充空白图像直到达到目标数量
        while len(all_images) < rows * cols:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            all_images.append(blank)
        
        # 创建网格图像
        grid = self.make_grid_image(all_images, grid_size)
        return grid

    def make_grid_image(self, images, grid_size):
        """创建图像网格"""
        rows, cols = grid_size
        cell_height, cell_width = images[0].shape[:2]
        grid = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)
        
        for idx, img in enumerate(images):
            i = idx // cols
            j = idx % cols
            grid[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = img
        
        return grid

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型（需要你的模型定义）
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    
    # 类别名称
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # 路径设置
    test_dir = '/home/ubuntu/IndustrialDigitDatasetGenerator/sigle_num'
    output_dir = '/home/ubuntu/IndustrialDigitDatasetGenerator/evaluation_results'
    
    # 创建评估器
    evaluator = ModelEvaluator(model, device, class_names, test_dir, output_dir)
    
    # 运行评估
    print("开始模型评估...")
    evaluator.evaluate_model()
    
    # 生成混淆矩阵
    print("生成混淆矩阵...")
    evaluator.plot_confusion_matrix()
    
    # 打印分类报告
    evaluator.print_classification_report()
    
    # 生成示例图片网格
    print("生成示例图片...")
    success_grid = evaluator.create_example_grid(evaluator.success_examples, True)
    failed_grid = evaluator.create_example_grid(evaluator.failed_examples, False)
    
    if success_grid is not None:
        cv2.imwrite(os.path.join(output_dir, 'success_examples.png'), 
                    cv2.cvtColor(success_grid, cv2.COLOR_RGB2BGR))
    if failed_grid is not None:
        cv2.imwrite(os.path.join(output_dir, 'failed_examples.png'), 
                    cv2.cvtColor(failed_grid, cv2.COLOR_RGB2BGR))
    
    print(f"评估完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()