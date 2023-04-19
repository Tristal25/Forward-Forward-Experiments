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
from argdata import argDict
import time


DATASETS = {
    'mnist': {
        'num_classes': 10,
        'img_size': 28,
        'num_channel': 1,
        'mean': (0.1307,),
        'std': (0.3081,),
    },

    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'num_channel': 3,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'num_channel': 3,
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

    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--lr', default=0.06, type=float)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--train_batch_size', default=10000, type=int)
    parser.add_argument('--test_batch_size', default=10000, type=int)

    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--norm', default=0, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--skip_connection', default=0, type=int)
    parser.add_argument('--unsupervised', default=0, type=int)
    parser.add_argument('--activation', type=str.lower,
                        choices=['relu','tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu'],
                        default='relu')
    parser.add_argument('--margin', default=0.1, type=float)
    parser.add_argument('--loss', type=str.lower,
                        choices=['p_pos', 'logistic'],
                        default='logistic')
    parser.add_argument('--neg_data', type=str.lower,
                        choices=['once', 'random'],
                        default='random')
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
    set_seed(args.random_seed)

    if args.device=='gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    print (f'config: {args}')

    num_classes = DATASETS[args.dataset]['num_classes']
    img_size=DATASETS[args.dataset]['img_size']
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
    elif args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

    print (f"Size of trainset: {len(trainset)}")
    print (f"Size of testset: {len(testset)}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    print (f"Size of trainloader: {len(trainloader)}")
    print (f"Size of testloader: {len(testloader)}")

    # create network
    net = ff.Net(DATASETS[args.dataset], device, args)

    # start training
    t0 = time.time()
    tot_num_batch=len(trainloader)

    for train_batch_idx, (images, target) in enumerate(trainloader):
        images, target = images.to(device), target.to(device)

        print (f"train_batch: [{train_batch_idx}|{tot_num_batch-1}]")
        net.train(images, target)

        train_acc = net.predict(images).eq(target).float().mean().item()

        # start evaluation
        with torch.no_grad():
            acc1_num_sum = 0
            num_input_sum = 0
            for val_batch_idx, (images, target) in enumerate(testloader):
                images, target = images.to(device), target.to(device)

                cur_val_acc = net.predict(images).eq(target).float().mean().item()

                acc1_num_sum += float(cur_val_acc * images.shape[0])
                num_input_sum += images.shape[0]

            print('train acc:', train_acc)
            print('test acc:', acc1_num_sum/num_input_sum)
    t1 = time.time()

    return acc1_num_sum/num_input_sum, t1-t0



    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='FF training script')
    # parser = init_parser(parser)
    # args = parser.parse_args()
    
    # load args from dictionary argDict
    args = argparse.Namespace(**argDict)
    trainOne = True

    if trainOne:
        train_top_module(args)
    else:
        hidden_sizes = [300, 400, 500]
        epochs = [60, 100, 300, 500, 1000]
        thresholds = np.arange(0.5, 10.5, 0.5)
        activations = ['relu','leaky_relu', 'elu']
        testacc = np.zeros((len(activations), len(thresholds)))
        total_time = np.zeros((len(activations), len(thresholds)))
        for i, hidden_size in enumerate(hidden_sizes):
            for j, epoch in enumerate(epochs):
                for l, activation in enumerate(activations):
                    for m, threshold in enumerate(thresholds):
                        args.activation = activation
                        args.threshold = threshold
                        args.hidden_size = hidden_size
                        args.epochs = epoch
                        testacc[l][m], total_time[l][m] = train_top_module(args)
                # plot the test accuracy
                for index, accuracies in enumerate(testacc):
                    plt.plot(thresholds, accuracies, marker='o', label=f"Activation: {activations[index]}")

                plt.xlabel('threshold')
                plt.ylabel('test accuracy')
                # range y axis from 0.8 to 1
                plt.ylim(0.8, 1)
                plt.title(f"test accuracy vs threshold vs activation (Hidden size: {hidden_size}, Epochs: {epoch})")
                plt.legend()
                plt.savefig(f"images/testacc2_{hidden_sizes[i]}_{epochs[j]}.png")
                plt.clf()