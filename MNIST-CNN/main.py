import argparse
import wandb
import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from src.train import train
from src.evaluate import test
from src.models.model import CNN
from src.utils.config import ROOT_PATH

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

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')

parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')

args = parser.parse_args()

wandb.init(project="MNIST")
wandb.watch_called = False
config = wandb.config 
config.architecture = "CNN"
config.dataset = "MNIST"
config.batch_size = args.batch_size
config.test_batch_size = args.test_batch_size
config.epochs = args.epochs
config.lr = args.lr
config.gamma = args.gamma
config.no_cuda = args.no_cuda
config.no_mps = args.no_mps
config.dry_run = args.dry_run
config.seed = args.seed
config.log_interval = args.log_interval
config.save_model = args.save_model

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
train_dataset = datasets.MNIST(root=ROOT_PATH+'/data/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=ROOT_PATH+'/data/', train=False, download=True, transform=transform)

# 数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = CNN().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

if args.save_model:
    torch.save(model.state_dict(), ROOT_PATH + "/saved_models/mnist_cnn.pt")