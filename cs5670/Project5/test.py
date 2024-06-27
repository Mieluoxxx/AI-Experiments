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
from net import AlexNet  # 从 net.py 导入 AlexNet

# 设置随机种子和精度
L.seed_everything(42)
torch.set_float32_matmul_precision("high")

# 数据转换
transform = transforms.Compose(
    [
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ]
)

# 数据目录
data_dir = "./data"

# 加载测试数据集
test_dataset = datasets.ImageFolder(data_dir + "/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=804, shuffle=False, num_workers=4)

# 加载最佳模型检查点
best_model_path = "checkpoints/best-checkpoint.ckpt"
best_model = AlexNet.load_from_checkpoint(best_model_path, num_classes=1)

# 设置模型为评估模式并将其移至 GPU
best_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model.to(device)

# 定义评估指标和损失函数
test_accuracy = BinaryAccuracy().to(device)
test_loss_fn = nn.BCEWithLogitsLoss()

# 用于保存分类错误的图像路径的列表
mis_cat = []
mis_dog = []

# 用于收集真实标签和预测标签的列表
true_labels = []
predicted_labels = []

# 禁用梯度计算以提高推理速度
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 进行预测
        outputs = best_model(images).squeeze(1)

        # 计算损失和准确率
        test_loss = test_loss_fn(outputs, labels.float())
        test_acc = test_accuracy(outputs, labels)

        # 收集分类错误的图像路径
        preds = (torch.sigmoid(outputs).cpu() > 0.5).numpy().astype(int)
        incorrect_indices = preds != labels.cpu().numpy()

        for idx, incorrect in enumerate(incorrect_indices):
            if incorrect:
                image_path = test_dataset.imgs[idx][0]
                if labels.cpu().numpy()[idx] == 0:  # 真实标签为猫
                    mis_cat.append(image_path)
                else:  # 真实标签为狗
                    mis_dog.append(image_path)

        # 收集真实标签和预测标签
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds)

        print(f"测试损失: {test_loss.item():.4f}, 测试准确率: {test_acc.item():.4f}")

# 计算混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    annot_kws={"fontsize": 15, "fontweight": "bold"},
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"],
)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
plt.title("confusion_matrix", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 保存图像文件
plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
plt.show()


# 绘制分类错误的猫和狗图像
def plot_images(cat_paths, dog_paths, n=5):
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(cat_paths[:n]):
        img = Image.open(img_path)
        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.title(f"Dog {i + 1}")
        plt.axis("off")
    for i, img_path in enumerate(dog_paths[:n]):
        img = Image.open(img_path)
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(img)
        plt.title(f"Cat {i + 1}")
        plt.axis("off")
    plt.savefig("results/mistake.png")
    plt.show()


print("显示分类错误的猫和狗图像:")
plot_images(mis_cat, mis_dog, n=5)
