import importlib
import importlib.util
import os
import sys
import torch
import torch.nn as nn


# Robustly load cfgs from config/vgg_cfg.py
def _load_vgg_cfgs():
    # preferred: standard import if repo root is on sys.path
    try:
        return importlib.import_module('config.vgg_cfg').cfgs
    except Exception:
        pass

    # search upward from this file for config/vgg_cfg.py
    cur = os.path.dirname(__file__)
    for _ in range(6):
        candidate = os.path.join(cur, 'config', 'vgg_cfg.py')
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location('config.vgg_cfg', candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.cfgs
        cur = os.path.dirname(cur)

    # try current working directory
    candidate = os.path.join(os.getcwd(), 'config', 'vgg_cfg.py')
    if os.path.isfile(candidate):
        spec = importlib.util.spec_from_file_location('config.vgg_cfg', candidate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.cfgs

    raise ImportError('Could not import vgg configs from config/vgg_cfg.py')


cfgs = _load_vgg_cfgs()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
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


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16(num_classes=1000, batch_norm=False):
    return VGG(make_layers(cfgs['VGG16'], batch_norm=batch_norm), num_classes=num_classes)
