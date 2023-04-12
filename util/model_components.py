import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import random

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
        #print("correct y:", y[0:10])
        y_fake = [random.choice(list(set(range(num_classes)) - set([item]))) for item in y.tolist()]
        #print("fake y:", y_fake[0:10])
        return overlay_y_on_x(x, y_fake, num_classes)


class Net(nn.Module):

    def __init__(self, dims, device, args):
        super().__init__()
        self.layers = [Layer(dims, args.hidden_size, args=args).to(device)]
        for d in range(args.num_layers - 1):
            self.layers += [Layer(args.hidden_size, args.hidden_size, args=args).to(device)]
        self.num_epochs = args.epochs
        #self.print_freq = args.print_freq
        self.norm = args.norm
        self.dropout = args.dropout
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.skip_connection = args.skip_connection

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

    def train(self, images, target, num_classes=10):
        for epoch in tqdm(range(self.num_epochs)):
            #if epoch % self.print_freq == 0:
                #print(f'epoch {epoch} ...')
            h_pos = generate_data(images, target, num_classes)
            h_neg = generate_data(images, target, num_classes, neg=True)
            for i, layer in enumerate(self.layers):
                #if epoch % self.print_freq == 0:
                    #print(f'training layer {i} ...')
                h_prev_pos, h_prev_neg = h_pos, h_neg
                h_pos, h_neg = layer.train(h_pos, h_neg)

                if self.norm:
                    h_pos = self.layer_norm(h_pos)
                    h_neg = self.layer_norm(h_neg)
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
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=args.lr)
        self.threshold = args.threshold
        self.norm = args.norm

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        g_pos = torch.mean(torch.pow(self.forward(x_pos), 2), 1)
        g_neg = torch.mean(torch.pow(self.forward(x_neg), 2), 1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -(g_pos - self.threshold),
            g_neg - self.threshold]))).mean()
        self.opt.zero_grad()
        # this backward just compute the local derivative on current layer
        # and hence is not considered backpropagation.
        loss.backward()
        self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()