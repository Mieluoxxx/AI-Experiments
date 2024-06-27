import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from net import AlexNet  # 从net.py导入AlexNet

L.seed_everything(42)
torch.set_float32_matmul_precision('high')

# 加载数据集
data_dir = './data'

# 定义数据集的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 定义LightningDataModule
class DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, transform):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        # 使用ImageFolder加载数据集
        self.train_dataset = datasets.ImageFolder(self.data_dir + '/train', transform=self.transform)
        self.val_dataset = datasets.ImageFolder(self.data_dir + '/val', transform=self.transform)
        self.test_dataset = datasets.ImageFolder(self.data_dir + '/test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

# 实例化数据模块和模型
data_module = DataModule(data_dir=data_dir, batch_size=32, transform=transform)
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
