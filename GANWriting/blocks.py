import torch
import torch.nn.functional as F
from torch import nn
import torch.autograd

#torch.autograd.set_detect_anomaly(True)

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features).to(device))
        self.beta = nn.Parameter(torch.zeros(num_features).to(device))

    def forward(self, x):
        x = x.to(device)
        mean = x.mean([1, 2, 3], keepdim=True)
        std = x.std([1, 2, 3], keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.gamma[:, None, None, None] * x + self.beta[:, None, None, None]

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x.to(device)
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = x.to(device)
        return self.model(x)

class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None, activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1, padding=1, pad_type='reflect', norm=norm, activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1, padding=1, pad_type='reflect', norm=norm, activation=activation, activation_first=True)
        
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1, padding=0, pad_type='reflect', norm=norm, activation='none', activation_first=True)

    def forward(self, x):
        x = x.to(device)
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim).to(device)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim).to(device)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False).to(device)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False).to(device)
        elif activation == 'tanh':
            self.activation = nn.Tanh().to(device)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = x.to(device)
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, pad_type='zero', norm='none', activation='relu', activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding).to(device)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding).to(device)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding).to(device)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim).to(device)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim).to(device)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim).to(device)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False).to(device)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False).to(device)
        elif activation == 'prelu':
            self.activation = nn.PReLU().to(device)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=False).to(device)
        elif activation == 'tanh':
            self.activation = nn.Tanh().to(device)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias).to(device)
        self.activation_first = activation_first

    def forward(self, x):
        x = x.to(device)
        if self.activation_first:
            if self.activation is not None:
                x = self.activation(x)
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if not self.activation_first:
            if self.activation is not None:
                x = self.activation(x)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features).to(device))
        self.bias = nn.Parameter(torch.zeros(num_features).to(device))
        self.register_buffer('running_mean', torch.zeros(num_features).to(device))
        self.register_buffer('running_var', torch.ones(num_features).to(device))

    def forward(self, x):
        x = x.to(device)
        assert self.weight is not None and self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

