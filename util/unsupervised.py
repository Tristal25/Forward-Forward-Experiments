import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import random
import math
import torch.nn.functional as F

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

def generate_unsup_neg(x, channels=1):
    mask = create_mask(int(math.sqrt(x.shape[1] / channels)), x.shape[0], channels, x.device)
    rand_ind = torch.randperm(x.shape[0])
    return x * mask + x[rand_ind] * (1 - mask)

class UnsupNet():

    def __init__(self, dataset, device, args, divided):
        # super().__init__()
        dims = dataset['num_channel']*dataset['img_size']*dataset['img_size']
        self.layers = [UnsupLayer(dims, args.hidden_size, args=args).to(device)]
        for d in range(args.num_layers - 1):
            self.layers += [UnsupLayer(args.hidden_size, args.hidden_size, args=args).to(device)]
        self.channels = dataset['num_channel']
        self.num_classes = dataset['num_classes']
        self.num_epochs = args.epochs
        self.dropout = args.dropout
        self.dropout_layer = nn.Dropout(p=args.dropout).to(device)
        self.skip_connection = args.skip_connection
        self.neg_data = args.neg_data
        self.divided = divided

    def predict(self, x):
        encoded = x
        for layer in self.layers:
            encoded = layer(encoded)
        return encoded
    
    def pairpos(self, h_pos, targets):
        h_pos2 = torch.zeros(h_pos.size())
        for i, t in enumerate(targets):
            h_pos2[i] = random.choice(self.divided[t])
        h_pos2 = torch.tensor(h_pos2).to("cuda")
        return h_pos2

    def train(self, images, target):
        for epoch in tqdm(range(self.num_epochs)):
            h_pos = images
            h_pos2 = self.pairpos(h_pos, target)
            h_neg = generate_unsup_neg(images, channels=self.channels)
            
            for i, layer in enumerate(self.layers):
                h_prev_pos, h_prev_neg = h_pos, h_neg
                h_pos, h_neg, h_pos2 = layer.train(h_pos, h_neg, h_pos2)

                if self.norm:
                    h_pos = self.layer_norm(h_pos).detach()
                    h_neg = self.layer_norm(h_neg).detach()
                if self.skip_connection and i > 0:
                    h_pos = h_pos + h_prev_pos
                    h_neg = h_neg + h_prev_neg
                if self.dropout != 0:
                    h_pos = self.dropout_layer(h_pos)
                    h_neg = self.dropout_layer(h_neg)

class UnsupLayer(nn.Linear):
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
        
    def Loss(self, x1, x2, x3, x4):
        sim_pos = F.cosine_similarity(x1, x2, dim=-1)
        sim_neg = F.cosine_similarity(x3, x4, dim=-1)
        loss = torch.clamp(1 - sim_pos.unsqueeze(1) + sim_neg.unsqueeze(0), min=0.0)
        return loss.mean()
    
    def train(self, x_pos, x_neg, x_pos2):
        # good pair
        x1_out = self.forward(x_pos)
        x2_out = self.forward(x_pos2)
        # neg pair
        x3_out = self.forward(x_pos)
        x4_out = self.forward(x_neg)
        
        loss = self.Loss(x1_out, x2_out, x3_out, x4_out)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return x1_out.detach(), x4_out.detach(), x2_out.detach()
    