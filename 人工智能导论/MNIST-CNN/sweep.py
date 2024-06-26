import argparse
import datetime
import os
import platform
import socket
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.datasets import MNIST

import wandb  

from src.models.model import CNN
from src.train import train
from src.evaluate import test


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')

parser.add_argument('--optim_type', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW', 'Adadelta'],
                    help='optimizer choice (default: SGD)')

parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')

parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# 初始化wandb，同时配置参数
nowtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(
    config=args,
    project="MNIST",
    notes=socket.gethostname(),
    save_code=True,
    name=nowtime
)

torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"device: {device}")


train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda and platform.system() != 'Windows':
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


# 加载训练集和测试集的张量
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# 创建MNIST数据集
train_dataset = MNIST(
    root='./data',
    train=True,
    download=True,  # 设置为False，因为您手动下载了数据集
    transform=transform
)

test_dataset = MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


sweep_config = {
    'method': 'random'  # grid bayes
    }

metric = {
    'name': 'test_accuracy',
    'goal': 'maximize'   
    }
sweep_config['metric'] = metric

sweep_config['parameters'] = {}

# 固定不变的超参
sweep_config['parameters'].update({
    'project':{'value':'MNIST'},
    'epochs': {'value': 10},
    'batch_size': {'value': 64},
    'test_batch_size': {'value': 1000},
    'no_cuda': {'value': False},
    'no_mps': {'value': False},
    'seed': {'value': 1},
    'log_interval': {'value': 10},
    })

# 离散型分布超参
sweep_config['parameters'].update({
    'optim_type': {
        'values': ['SGD', 'Adam', 'AdamW', 'Adadelta']
        },
    })

# 连续型分布超参
sweep_config['parameters'].update({
    
    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-4,
        'max': 1
      },
    'gamma': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.9
      },
})

sweep_id = wandb.sweep(sweep_config, project="MNIST")

def train():
    # 数据加载器
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
    model = CNN().to(device)
    model.run_id = wandb.run.id
    model.best_metrics = -1.0

    optimizer = torch.optim.__dict__[args.optim_type](params=model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        scheduler.step()
        if test_acc > model.best_metrics:
            model.best_metrics = test_acc
            torch.save(model.state_dict(), os.getcwd() + "/saved_models/mnist_cnn.pt")

        wandb.log({'epoch': epoch, 'test_acc': test_acc, 'best_test_acc': model.best_metrics})

    wandb.finish()


wandb.agent(sweep_id, train, count=20)