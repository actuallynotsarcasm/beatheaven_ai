import torch
import torch.nn as nn
import os
import time
from collections import OrderedDict


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        prefix = 'check_points/' + self.model_name +name+ '/'
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        print('model name', name.split('/')[-1] )
        torch.save(self.state_dict(), name)
        torch.save(self.state_dict(), prefix+'latest.pth')
        return name
    
    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def load_latest(self, notes):
        path = 'check_points/' + self.model_name +notes+ '/latest.pth'
        self.load_state_dict(torch.load(path))


class CQTNet(BasicModule):
    def __init__(self, emb_size=300):
        super().__init__()
        self.emb_size = emb_size
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 32, kernel_size=(12, 3), dilation=(1, 1), padding=(6, 0), bias=False)),
            ('norm0', nn.BatchNorm2d(32)), ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(32, 64, kernel_size=(13, 3), dilation=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv2', nn.Conv2d(64, 64, kernel_size=(13, 3), dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)), ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv4', nn.Conv2d(64, 128, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)), ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)), ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv6', nn.Conv2d(128, 256, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)), ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)), ('relu7', nn.ReLU(inplace=True)),
            ('pool7', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv8', nn.Conv2d(256, 512, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)), ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)), ('relu9', nn.ReLU(inplace=True)),
        ]))
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc0 = nn.Linear(512, emb_size)

    def forward(self, song1, samples1=None, song2=None, samples2=None):
        inputs = list(filter(lambda x: x is not None, [song1, samples1, song2, samples2]))
        outputs = []
        for x in inputs:
            shape = x.shape
            if len(shape) == 3:
                x = x.view(1, *shape)
            elif len(shape) == 5:
                x = x.view(-1, *shape[2:])
            N = x.size()[0]
            x = self.features(x)  # [N, 512, 57, 2~15]
            x = self.pool(x)
            x = x.view(N, -1)
            feature = self.fc0(x)
            if len(shape) == 5:
                feature = feature.view(*shape[:2], -1)
            outputs.append(feature)
        return tuple(outputs)