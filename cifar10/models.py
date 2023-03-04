import torch
import torch.nn as nn
from collections import OrderedDict

relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]

class VGGNet(nn.Module):
    def __init__(self, cfg, num_classes=10, mask_nums=None):
        super(VGGNet, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * len(cfg)
        self.relucfg = relucfg
        self.covcfg = convcfg
        self.features = self._make_layers(cfg)
        self.avgpool = nn.AvgPool2d(2)
        last_output_channels = cfg[-1] - self.mask_nums[-1]
        self.classifier = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(last_output_channels, last_output_channels)),
            ("norm", nn.BatchNorm1d(last_output_channels)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(last_output_channels, num_classes)),
        ]))
    
    def _make_layers(self, cfg):
        layers = nn.Sequential()
        in_channels, cnt = 3, 0
        for i, x in enumerate(cfg):
            if x == "M":
                layers.add_module(f"pool{i}", nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(x - self.mask_nums[cnt])
                cnt += 1
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                layers.add_module(f"conv{i}", conv2d)
                layers.add_module(f"norm{i}", nn.BatchNorm2d(out_channels))
                layers.add_module(f"relu{i}", nn.ReLU(inplace=True))
                in_channels = out_channels
        return layers
    
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, downsample_out_channels)
        self.bn2 = nn.BatchNorm2d(downsample_out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, residualblock, blocks, num_classes=10, mask_nums=None):
        super(ResNet, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * (sum(blocks) + 3)
        self.cnt = 0
        self.in_channels = 16 - self.mask_nums[0]
        self.conv = conv3x3(3, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(residualblock, 16, self.in_channels, blocks[0])
        self.layer2 = self._make_layer(residualblock, 32, 32-self.mask_nums[self.cnt+2], blocks[1], 2)
        self.layer3 = self._make_layer(residualblock, 64, 64-self.mask_nums[self.cnt+2], blocks[2], 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64-self.mask_nums[-(blocks[-1])], num_classes)
    
    def _make_layer(self, residualblock, out_channels, downsample_out_channels, blocks, stride=1):
        downsample = nn.Sequential(
            conv3x3(self.in_channels, downsample_out_channels, stride=stride),
            nn.BatchNorm2d(downsample_out_channels)) if stride != 1 else None
        layers = []
        self.cnt += 1
        layers.append(residualblock(self.in_channels, out_channels-self.mask_nums[self.cnt],
                                    downsample_out_channels, stride, downsample))
        self.in_channels = downsample_out_channels
        self.cnt = self.cnt + 1 if downsample else self.cnt
        for i in range(1, blocks):
            self.cnt += 1
            layers.append(residualblock(self.in_channels, out_channels-self.mask_nums[self.cnt], downsample_out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0):
        super(DenseBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = nn.functional.dropout(out, p=self.drop_rate, training=self.training) if self.drop_rate > 0 else out
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = nn.functional.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, denseblock, transition, depth, num_classes=10,
                 drop_rate=0, growth_rate=12, compression=1, mask_nums=None):
        super(DenseNet, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * depth
        self.blocks = (depth - 4) // 3
        self.drop_rate = drop_rate
        self.growth_rate = growth_rate
        
        self.in_channels = growth_rate * 2 - self.mask_nums[0]
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(denseblock, self.mask_nums[1:self.blocks+1])
        self.trans1 = self._make_transition(transition, 0, compression, self.mask_nums[self.blocks+1])
        self.dense2 = self._make_denseblock(denseblock, self.mask_nums[self.blocks+2:2*self.blocks+2])
        self.trans2 = self._make_transition(transition, 1, compression, self.mask_nums[2*self.blocks+2])
        self.dense3 = self._make_denseblock(denseblock, self.mask_nums[2*self.blocks+3:3*self.blocks+3])
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.in_channels, num_classes)
    
    def _make_denseblock(self, block, mask_nums):
        layers = []
        for i in range(self.blocks):
            layers.append(block(self.in_channels, out_channels=self.growth_rate-mask_nums[i], drop_rate=self.drop_rate))
            self.in_channels += self.growth_rate - mask_nums[i]
        return nn.Sequential(*layers)
    
    def _make_transition(self, transition, order, compression, mask_num):
        in_channels = self.in_channels
        out_channels = int((self.growth_rate*2+(order+1)*self.blocks*self.growth_rate-mask_num) * compression)
        self.in_channels = out_channels
        return transition(in_channels, out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def vgg16(mask_nums=None):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]
    return VGGNet(cfg=cfg, mask_nums=mask_nums)

def resnet56(mask_nums=None):
    return ResNet(ResidualBlock, [9, 9, 9], mask_nums=mask_nums)

def resnet110(mask_nums=None):
    return ResNet(ResidualBlock, [18, 18, 18], mask_nums=mask_nums)

def densenet40(mask_nums=None):
    return DenseNet(DenseBlock, Transition, 40, mask_nums=mask_nums)