import os
import torch
from torch.utils.data import Dataset
import torchvision.datasets.mnist as mnist
import gzip
import shutil
from config import ROOT_PATH

# gunzip *.gz   先解压文件

class MyMnist(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.data, self.targets = mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')), mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        else:
            self.data, self.targets = mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')), mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # 可能需要将图像和标签转换为PyTorch张量
        img = torch.from_numpy(img.numpy())
        target = torch.tensor(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

root_path = ROOT_PATH + '/data/raw/'
process_path = ROOT_PATH + '/data/processed/'

# 创建训练集和测试集的实例
train_dataset = MyMnist(root_path, train=True)
test_dataset = MyMnist(root_path, train=False)

# 保存为.pt文件
torch.save(train_dataset, process_path+'training.pt')
torch.save(test_dataset, process_path+'test.pt')