import math
import torch.nn as nn
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


class Scaler(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        output = x / self.ratio if self.training else x
        return output


class CNN(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1.0, track=cfg['norm_track']):
        super().__init__()
        if cfg['norm'] == 'bn':
            norm = nn.BatchNorm2d(hidden_size[0], momentum=0, track_running_stats=track)
        elif cfg['norm'] == 'in':
            norm = nn.GroupNorm(hidden_size[0], hidden_size[0])
        elif cfg['norm'] == 'ln':
            norm = nn.GroupNorm(1, hidden_size[0])
        elif cfg['norm'] == 'gn':
            norm = nn.GroupNorm(4, hidden_size[0])
        elif cfg['norm'] == 'none':
            norm = nn.Identity()
        else:
            raise ValueError('Not valid norm')

        if cfg['scale']:
            scaler = Scaler(rate)
        else:
            scaler = nn.Identity()

        blocks = [
            nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
            scaler,
            norm,
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2)
        ]

        for i in range(len(hidden_size) - 1):
            if cfg['norm'] == 'bn':
                norm = nn.BatchNorm2d(hidden_size[i + 1], track_running_stats=track)
            elif cfg['norm'] == 'in':
                norm = nn.GroupNorm(hidden_size[i + 1], hidden_size[i + 1])
            elif cfg['norm'] == 'ln':
                norm = nn.GroupNorm(1, hidden_size[i + 1])
            elif cfg['norm'] == 'gn':
                norm = nn.GroupNorm(4, hidden_size[i + 1])
            elif cfg['norm'] == 'none':
                norm = nn.Identity()
            else:
                raise ValueError('Not valid norm')

            if cfg['scale']:
                scaler = Scaler(rate)
            else:
                scaler = nn.Identity()

            blocks.extend([
                nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                scaler,
                norm,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        blocks = blocks[:-1]

        blocks.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size[-1], classes_size)
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        output = self.blocks(x)
        return output


def create_cnn(model_ratio=1.0, track=False):
    data_shape = cfg['data_shape']
    hidden_size = [math.ceil(model_ratio * x) for x in cfg['cnn_hidden_size']]
    classes_size = cfg['classes_size']
    scaler_rate = model_ratio
    model = CNN(data_shape, hidden_size, classes_size, scaler_rate, track)
    model.apply(init_param)
    return model
