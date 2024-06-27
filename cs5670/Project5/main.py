import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import lightning as L
from torchmetrics.classification import BinaryAccuracy
from lightning.pytorch.callbacks import ModelCheckpoint
L.seed_everything(42)
torch.set_float32_matmul_precision('high') 


# 加载数据集
data_dir = './data'

# 定义数据集的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


#######################################
########验证数据集和数据加载器############
#######################################

# # 构建数据加载器
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # 确认数据加载器工作正常
# for images, labels in train_loader:
#     print(images.shape, labels.shape)
#     break

#######################################
########验证数据集和数据加载器############
#######################################

# 使用LightningModule和Trainer进行训练
class AlexNet(L.LightningModule):
    def __init__(self, num_classes=1):  
        super(AlexNet, self).__init__()
        self.save_hyperparameters()
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
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
        acc = self.train_accuracy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
        acc = self.val_accuracy(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

# 定义LightningDataModule
class DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, transform):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        # 使用ImageFolder加载数据集
        self.train_dataset = datasets.ImageFolder(data_dir+'/train', transform=transform)
        self.val_dataset = datasets.ImageFolder(data_dir+'/val', transform=transform)
        self.test_dataset = datasets.ImageFolder(data_dir+'/test', transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

# 实例化数据模块和模型
data_module = DataModule(data_dir='./data', batch_size=32, transform=transform)
model = AlexNet(num_classes=1)

# 定义 ModelCheckpoint 回调
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',  # 监控的指标
    dirpath='checkpoints',  # 保存的路径
    filename='best-checkpoint',  # 保存的文件名
    save_top_k=1,  # 仅保存最好的模型
    mode='max'  # 指标越大越好
)

# 使用Trainer进行训练
trainer = L.Trainer(
    max_epochs=15,
    accelerator='gpu',
    devices=1,
    callbacks=[checkpoint_callback],
)
trainer.fit(model, datamodule=data_module)

# 训练完成后，可以加载最佳模型
best_model_path = checkpoint_callback.best_model_path
print(f"Best model saved at: {best_model_path}")
