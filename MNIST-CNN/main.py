import argparse
import socket
import wandb
import torch
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from src.train import train
from src.evaluate import test
from src.models.model import CNN
from torchvision.datasets import MNIST
from src.utils.config import ROOT_PATH
import datetime

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')

parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')

parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')

parser.add_argument('--no-cuda', action='store_true', default=True,
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

# 直接使用wandb.config配置参数
wandb.watch_called = False
wandb.config.dataset = "MNIST"


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
if use_cuda:
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
    download=False,  # 设置为False，因为您手动下载了数据集
    transform=transform
)

test_dataset = MNIST(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

# 数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


model = CNN().to(device)
model.run_id = wandb.run.id
model.best_metrics = -1.0
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test_acc = test(model, device, test_loader)
    scheduler.step()
    if test_acc > model.best_metrics:
        model.best_metrics = test_acc
        torch.save(model.state_dict(), ROOT_PATH + "/saved_models/mnist_cnn.pt")

    wandb.log({'epoch': epoch, 'test_acc': test_acc, 'best_test_acc': model.best_metrics})

wandb.finish()