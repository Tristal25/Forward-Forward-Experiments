import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import random
import math
import torch.nn.functional as F

def overlay_y_on_x(x, y, num_classes=10):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :num_classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def generate_data(x, y, num_classes=10, neg = False):
    if not neg:
        return overlay_y_on_x(x, y, num_classes)
    else:
        y_fake = [random.choice(list(set(range(num_classes)) - set([item]))) for item in y.tolist()]
        return overlay_y_on_x(x, y_fake, num_classes)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class Net():
    def __init__(self, dataset, device, args):
        # super().__init__()
        dims = dataset['num_channel']*dataset['img_size']*dataset['img_size']
        self.layers = [Layer(dims, args.hidden_size, args=args).to(device)]
        for d in range(args.num_layers - 1):
            self.layers += [Layer(args.hidden_size, args.hidden_size, args=args).to(device)]
        self.channels = dataset['num_channel']
        self.num_classes = dataset['num_classes']
        self.num_epochs = args.epochs
        self.dropout = args.dropout
        self.dropout_layer = nn.Dropout(p=args.dropout).to(device)
        self.skip_connection = args.skip_connection
        self.neg_data = args.neg_data

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, images, target):
        if self.neg_data != 'random':
            h_pos_orig = generate_data(images, target, self.num_classes, False)
            h_neg_orig = generate_data(images, target, self.num_classes, True)
        for epoch in tqdm(range(self.num_epochs)):
            if self.neg_data == 'random':
                h_pos = generate_data(images, target, self.num_classes, False)
                h_neg = generate_data(images, target, self.num_classes, True)
            else:
                h_pos = h_pos_orig
                h_neg = h_neg_orig
            for i, layer in enumerate(self.layers):
                h_prev_pos, h_prev_neg = h_pos, h_neg
                h_pos, h_neg = layer.train(h_pos, h_neg)

                if self.skip_connection and i > 0:
                    h_pos = h_pos + h_prev_pos
                    h_neg = h_neg + h_prev_neg
                if self.dropout != 0:
                    h_pos = self.dropout_layer(h_pos)
                    h_neg = self.dropout_layer(h_neg)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None, args=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.opt = Adam(self.parameters(), lr=args.lr)
        self.threshold = args.threshold
        self.margin = args.margin
        self.norm = args.norm
        self.loss = args.loss
        self.activation = self.getActivation(args.activation)

    def getActivation(self, activation):
        if activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'tanh':
            return torch.nn.Tanh()
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation == 'elu':
            return torch.nn.ELU()
        elif activation == 'gelu':
            return torch.nn.GELU()
        else:
            return torch.nn.LeakyReLU()

    def forward(self, x):
        if self.norm:
            x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.activation(
            torch.mm(x, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        pos_out = self.forward(x_pos)
        neg_out = self.forward(x_neg)

        if self.loss == 'p_pos':
            g_pos = torch.sum(torch.pow(pos_out, 2), 1)
            g_neg = torch.sum(torch.pow(neg_out, 2), 1)
            loss = ((self.threshold + self.margin - sigmoid(g_pos)) + \
                    (sigmoid(g_neg) - self.threshold + self.margin)).mean()
        else:
            g_pos = torch.mean(torch.pow(pos_out, 2), 1)
            g_neg = torch.mean(torch.pow(neg_out, 2), 1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -(g_pos - self.threshold + self.margin),
                g_neg - self.threshold - self.margin]))).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return pos_out.detach(), neg_out.detach()