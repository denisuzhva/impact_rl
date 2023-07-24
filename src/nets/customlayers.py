import numpy as np
import torch
from torch import nn
from src.nets.utils_dcnn import calc_conv2d_shape



def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['softsign', nn.Softsign()],
        ['gelu', nn.GELU()],
        ['sigmoid', nn.Sigmoid()],
        ['tanh', nn.Tanh()],
        ['none', nn.Identity()]
    ])[activation]


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)


class ResidualBlock2d(nn.Module):
    """2 dimensional residual block."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 act='relu', last_act='relu'):
        super().__init__()
        self.__conv1 = conv3x3(in_channels, out_channels, stride)
        self.__bn1 = nn.BatchNorm2d(out_channels)
        self.__act = activation_func(act)
        self.__conv2 = conv3x3(out_channels, out_channels)
        self.__bn2 = nn.BatchNorm2d(out_channels)
        self.__downsample = downsample
        self.__last_act = activation_func(last_act)

    def forward(self, x):
        residual = x
        out = self.__conv1(x)
        out = self.__bn1(out)
        out = self.__act(out)
        out = self.__conv2(out)
        out = self.__bn2(out)
        if self.__downsample:
            residual = self.__downsample(x)
        out += residual
        out = self.__last_act(out)
        return out


class DenseBlock2d(nn.Module):
    """2 dimensional dense block."""

    def __init__(self, in_channels, out_channels, n_denselayers, act='relu'):    
        super().__init__()
        self.__n_denselayers = n_denselayers
        self.__act = activation_func(act)
        self.__bn = nn.BatchNorm2d(in_channels)
        self.__conv_init = conv3x3(in_channels, out_channels)
        self.__conv_denselist = nn.ModuleList([conv3x3(out_channels * (idx+1), out_channels) 
                                               for idx in range(n_denselayers)])
    
    def forward(self, x):
        bn = self.__bn(x) 
        conv_regulars = []
        conv_regulars.append(self.__act(self.__conv_init(x)))
        cdense = conv_regulars[0]

        for idx in range(self.__n_denselayers):
            conv_regulars.append(self.__act(self.__conv_denselist[idx](cdense)))
            cdense = torch.cat(conv_regulars, 1)
        
        return cdense


class ConvCS2d(nn.Module):
    """2-dimensional convolutional compressed sensing module with fixed randomized weights."""

    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding,
                 val_distrib='gauss', weight=None, rnd_seed=0):
        """
        Args:
            in_channels:    Number of input channels
            out_channels:   Number of output channels
            kernel_size:    Number (if equal for both dimensions) or tuple of kernel size 
            stride:         Number (if equal for both dimensions) or tuple of kernel stride along two dimensions
            padding:        Number (if equal for both dimensions) or tuple of padding size for two dimensions 
            val_distrib:    Distribution of the convolution weight values
                custom:     Custom weights
                gauss:      Normal distrubution with zero mean and 1/sqrt(m) std (m is the compressed dimensionality)
                b2:         Bernoulli distribution (0 or 1 with probability 1/2)
                bdiff:      Bernoulli distribution (0 or 1 with probability 1/2) with inverted channels
            weight:         Custom weights
            rnd_seed:       Random seed
        """
        super().__init__()
        self.__convCS2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        with torch.no_grad():
            if val_distrib == 'custom':
                self.__convCS2d.weight = nn.Parameter(weight)
            elif val_distrib == 'gauss':
                nn.init.normal_(self.__convCS2d.weight, mean=0, std=1)
            elif val_distrib == 'b2':
                probs = torch.full((out_channels, in_channels, kernel_size, kernel_size), 0.5)
                weight_b = torch.bernoulli(probs)
                self.__convCS2d.weight = nn.Parameter(weight_b)
            elif val_distrib == 'bdiff':
                half_shape = (out_channels//2, in_channels, kernel_size, kernel_size)
                ones = torch.full(half_shape, 1.)
                probs = torch.full(half_shape, 0.5)
                init_weights = torch.bernoulli(probs)
                inv_weights = ones - init_weights
                self.__convCS2d.weight[:out_channels//2] = nn.Parameter(init_weights)
                self.__convCS2d.weight[out_channels//2:] = nn.Parameter(inv_weights)
            elif val_distrib == 'b3':
                pass
            elif val_distrib == 'bb':
                pass
            else:
                raise NotImplementedError("Unknown distribution name %s" % val_distrib)

        self.__convCS2d.requires_grad_(False)

    def forward(self, x):
        out = self.__convCS2d(x)
        return out

    def get_weigths(self):
        return self.__convCS2d.weight


