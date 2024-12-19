import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=1, batch_norm=False):
    layers = []
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def adapt_pretrained_weights(weights, in_channels):
    conv1_weight = weights['features.0.weight']
    if in_channels == 1:
        conv1_weight = conv1_weight.mean(dim=1, keepdim=True)
    weights['features.0.weight'] = conv1_weight
    return weights

def vgg19_bn(pretrained=False, in_channels=1, **kwargs):
    model = VGG(make_layers(cfg['E'], in_channels=in_channels, batch_norm=True), **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        total_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        total_dict = adapt_pretrained_weights(total_dict, in_channels)
        partial_dict = {k: v for k, v in total_dict.items() if k in model_dict}
        model_dict.update(partial_dict)
        model.load_state_dict(model_dict)
    return model

# Test the VGG model with single channel input
if __name__ == "__main__":
    model = vgg19_bn(pretrained=False, in_channels=1)
    x = torch.randn(128, 1, 128, 128)  # Example input with 1 channel and size 128x128
    output = model(x)
    print(f"Output shape: {output.shape}")
