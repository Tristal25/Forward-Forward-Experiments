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



    #train_loader, test_loader = MNIST_loaders()

    net = ff.Net([DATASETS[args.dataset]['num_channel']*img_size*img_size, 100, 100], device, args)  # 2 linear layers with 784=>500, 500=>500

    # start training


        #net.train(x_pos, x_neg)



    tot_num_batch=len(trainloader)

    for train_batch_idx, (images, target) in enumerate(trainloader):
        images, target = images.to(device), target.to(device)

        print (f"train_batch: [{train_batch_idx}|{tot_num_batch-1}]")

        for layeridx, curlayer in enumerate(net.layers):


            if layeridx == 0:  # only first layer get data, otherwise get previous forward data
                #print (f"images.shape: {images.shape}")

                x_pos = ff.overlay_y_on_x(images, target)

                rnd = torch.randperm(images.size(0))
                x_neg = ff.overlay_y_on_x(images, target[rnd])

            for epochidx in range(args.epochs):
                if epochidx%args.print_freq==0:
                    print (f"train_batch: [{train_batch_idx}|{tot_num_batch-1}] Layer: [{layeridx:}] Epoch: [{epochidx}]")


                #g_pos = self.forward(x_pos).pow(2).mean(1)  # take mean on the axis=1
                # output is [50000, 500] => [50000, 1]
                g_pos = torch.mean(torch.pow(curlayer.forward(x_pos), 2), 1)
                #print (f"g_pos:{g_pos.shape}")

                #g_neg = self.forward(x_neg).pow(2).mean(1)
                # output is [50000, 500] => [50000, 1]
                g_neg = torch.mean(torch.pow(curlayer.forward(x_neg), 2), 1)

                # The following loss pushes pos (neg) samples to
                # values larger (smaller) than the self.threshold.

                # exp is to ensure positive
                loss = torch.log(1 + torch.exp(torch.cat([
                    -(g_pos - args.threshold),
                    g_neg - args.threshold]))).mean()

                curlayer.opt.zero_grad()
                # this backward just compute the local derivative on current layer
                # and hence is not considered backpropagation.
                loss.backward()
                curlayer.opt.step()

            # finished training on curlayer, pass to next layer
            x_pos, x_neg=curlayer.forward(x_pos).detach(), curlayer.forward(x_neg).detach()


        train_acc=net.predict(images).eq(target).float().mean().item()



        # start evaluation
        with torch.no_grad():
            acc1_num_sum = 0
            num_input_sum=0
            for val_batch_idx, (images, target) in enumerate(testloader):
                images, target = images.to(device), target.to(device)

                cur_val_acc=net.predict(images).eq(target).float().mean().item()

                acc1_num_sum += float(cur_val_acc * images.shape[0])
                num_input_sum+=images.shape[0]



            print('train acc:',train_acc)
            print('test acc:', acc1_num_sum/num_input_sum)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FF training script')
    parser = init_parser(parser)
    args = parser.parse_args()

    train_top_module(args)

