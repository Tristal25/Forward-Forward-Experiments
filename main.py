import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import util.model_components as ff
import argparse
import random
import numpy as np


DATASETS = {
    'mnist': {
        'num_classes': 10,
        'img_size': 28,
        'mean': (0.1307,),
        'std': (0.3081,),
    },

    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
    }
}

def init_parser(parser):

    parser.add_argument('--device', type=str, default='gpu')

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['mnist','cifar10', 'cifar100'],
                        default='mnist')

    parser.add_argument('--threshold', default=2.0, type=float)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--epochs', default=1000, type=int)

    parser.add_argument('--train_batch_size', default=60000, type=int)
    parser.add_argument('--test_batch_size', default=10000, type=int)
    return parser


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader







    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train_top_module(args):
    set_seed(42)

    if args.device=='gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    print (f'config: {args}')

    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']
    transform = Compose([
        ToTensor(),
        Normalize(img_mean, img_std),
        Lambda(lambda x: torch.flatten(x))])


    if args.dataset=='mnist':
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)

    print (f"Size of trainset: {len(trainset)}")
    print (f"Size of testset: {len(testset)}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    print (f"Size of trainloader: {len(trainloader)}")
    print (f"Size of testloader: {len(testloader)}")



    #train_loader, test_loader = MNIST_loaders()

    net = ff.Net([784, 500, 500], device, args)  # 2 linear layers with 784=>500, 500=>500

    # start training
    for batch_idx, (images, target) in enumerate(trainloader):
        images, target = images.to(device), target.to(device)

        x_pos = ff.overlay_y_on_x(images, target)

        rnd = torch.randperm(images.size(0))
        x_neg = ff.overlay_y_on_x(images, target[rnd])

        net.train(x_pos, x_neg)

        print('train error:', 1.0 - net.predict(images).eq(target).float().mean().item())
        print('train acc:', net.predict(images).eq(target).float().mean().item())


    # start evaluation
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(testloader):
            images, target = images.to(device), target.to(device)


            print('test error:', 1.0 - net.predict(images).eq(target).float().mean().item())
            print('test acc:', net.predict(images).eq(target).float().mean().item())


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FF training script')
    parser = init_parser(parser)
    args = parser.parse_args()

    train_top_module(args)

