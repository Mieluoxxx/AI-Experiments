import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAccuracy
import lightning as L
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置随机种子和精度
L.seed_everything(42)
torch.set_float32_matmul_precision('high')

# 数据转换
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

# 数据目录
data_dir = './data'

# 加载测试数据集
test_dataset = datasets.ImageFolder(data_dir + '/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 定义模型
class AlexNet(L.LightningModule):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# 加载最佳模型
best_model_path = 'checkpoints/best-checkpoint.ckpt'
best_model = AlexNet.load_from_checkpoint(best_model_path, num_classes=1)

# 设置评估模式并将模型移至GPU
best_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model.to(device)

# 定义评估指标和损失函数
test_accuracy = BinaryAccuracy().to(device)
test_loss_fn = nn.BCEWithLogitsLoss()

# 用于保存分类错误的图像地址
mis_cat = []
mis_dog = []

# 用于收集 true_label 和 predict_label
true_labels = []
predicted_labels = []

# 禁用梯度计算以提高推理速度
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 进行预测
        outputs = best_model(images).squeeze(1)
        
        # 计算损失和准确度
        test_loss = test_loss_fn(outputs, labels.float())
        test_acc = test_accuracy(outputs, labels)
        
        # 获取分类错误的图像地址
        preds = (torch.sigmoid(outputs).cpu() > 0.5).numpy().astype(int)
        incorrect_indices = (preds != labels.cpu().numpy())
        
        for idx, incorrect in enumerate(incorrect_indices):
            if incorrect:
                image_path = test_dataset.imgs[idx][0]
                if labels.cpu().numpy()[idx] == 0:  # 真实标签为猫
                    mis_cat.append(image_path)
                else:  # 真实标签为狗
                    mis_dog.append(image_path)
                
        # 收集 true_label 和 predict_label
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds)
        
        print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_acc.item():.4f}")

# Compute Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={'fontsize': 15, 'fontweight': 'bold'},
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot as an image file
plt.savefig('results/confusion_matrix.png', bbox_inches='tight')
plt.show()


# 绘制分类错误的猫和狗图像在同一张图上
def plot_images(cat_paths, dog_paths, n=5):
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(cat_paths[:n]):
        img = Image.open(img_path)
        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.title(f"Dog {i + 1}")
        plt.axis('off')
    for i, img_path in enumerate(dog_paths[:n]):
        img = Image.open(img_path)
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(img)
        plt.title(f"Cat {i + 1}")
        plt.axis('off')
    plt.savefig('results/mistake.png')
    plt.show()

print("Displaying misclassified cats and dogs:")
plot_images(mis_cat, mis_dog, n=5)
