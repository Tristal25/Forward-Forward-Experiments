import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(nn.Module):

    def __init__(self, dims, device, args):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], args=args).to(device)]

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

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print(f'training layer {i} ...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None, args=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=args.lr)
        self.threshold = args.threshold
        self.num_epochs = args.epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            #g_pos = self.forward(x_pos).pow(2).mean(1)  # take mean on the axis=1
            # output is [50000, 500] => [50000, 1]
            g_pos = torch.mean(torch.pow(self.forward(x_pos), 2), 1)
            #print (f"g_pos:{g_pos.shape}")

            #g_neg = self.forward(x_neg).pow(2).mean(1)
            # output is [50000, 500] => [50000, 1]
            g_neg = torch.mean(torch.pow(self.forward(x_neg), 2), 1)

            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.

            # exp is to ensure positive
            loss = torch.log(1 + torch.exp(torch.cat([
                -(g_pos - self.threshold),
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the local derivative on current layer
            # and hence is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()