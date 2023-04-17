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

def create_mask(size, batch_size, channels=1, device=None, num_blurs=10):
    # image = torch.randint(low=0, high=2, size=(channels, size, size)).float()
    image = torch.rand(batch_size, channels, size, size).to(device)
    
    kernel_base = torch.tensor([1/4, 1/2, 1/4]).float()
    kernel = kernel_base.repeat(channels, 1, 1, 1).view(channels, 1, 1, 3).to(device)
    for i in range(num_blurs):
        image = F.conv2d(image, kernel, padding=(0, 1), groups=channels)
        image = F.conv2d(image, kernel.transpose(2, 3), padding=(1, 0), groups=channels)

    mask = (image > 0.5).float()
    mask = torch.flatten(mask, start_dim=1)
    return mask

def generate_data(x, y=None, num_classes=10, neg = False, channels=1):
    if y is None:
        if not neg:
            return x
        else:
            mask = create_mask(int(math.sqrt(x.shape[1] / channels)), x.shape[0], channels, x.device)
            rand_ind = torch.randperm(x.shape[0])
            return x * mask + x[rand_ind] * (1 - mask)

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
        self.norm = args.norm
        self.dropout = args.dropout
        self.layer_norm = nn.LayerNorm(args.hidden_size).to(device)
        self.dropout_layer = nn.Dropout(p=args.dropout).to(device)
        self.skip_connection = args.skip_connection
        self.unsupervised = args.unsupervised
        self.device = device

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]

            #print (len(goodness), goodness)
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, images, target):
        if self.unsupervised:
            h_pos = generate_data(images, neg=False, channels=self.channels)
            h_neg = generate_data(images, neg=True, channels=self.channels)
        else:
            h_pos = generate_data(images, target, self.num_classes, False, self.channels)
            h_neg = generate_data(images, target, self.num_classes, True, self.channels)
        for i, layer in enumerate(self.layers):
            h_prev_pos, h_prev_neg = h_pos, h_neg
            h_pos, h_neg = layer.train(h_pos, h_neg)

            if self.norm:
                h_pos = self.layer_norm(h_pos).detach()
                h_neg = self.layer_norm(h_neg).detach()
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
        self.num_epochs = args.epochs
        if args.activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif args.activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif args.activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif args.activation == 'elu':
            self.activation = torch.nn.ELU()
        else:
            self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.activation(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for epoch in tqdm(range(self.num_epochs)):
            pos_out = self.forward(x_pos)
            neg_out = self.forward(x_neg)

            g_pos = torch.mean(torch.pow(pos_out, 2), 1)
            g_neg = torch.mean(torch.pow(neg_out, 2), 1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            #loss = ((self.threshold + self.margin - sigmoid(g_pos)) + \
            #       (sigmoid(g_neg) - self.threshold + self.margin)).mean()

            loss = torch.log(1 + torch.exp(torch.cat([
                -(g_pos - self.threshold + self.margin),
                g_neg - self.threshold - self.margin]))).mean()
            self.opt.zero_grad()
            # this backward just compute the local derivative on current layer
            # and hence is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()