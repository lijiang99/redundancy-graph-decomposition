import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class VGGNet(nn.Module):
    def __init__(self, cfg, num_classes=10, mask_nums=None):
        super(VGGNet, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * len(cfg)
        self.features = self._make_layers(cfg)
        self.avgpool = nn.AvgPool2d(2)
        last_output_channels = cfg[-1] - self.mask_nums[-1]
        self.classifier = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(last_output_channels, cfg[-1])),
            ("norm", nn.BatchNorm1d(cfg[-1])),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(cfg[-1], num_classes)),
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

class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5_1, n5x5_2, pool_channels, mask_nums):
        super(Inception, self).__init__()
        
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1-mask_nums[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(n1x1-mask_nums[0]),
            nn.ReLU(True),
        )
        
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduce-mask_nums[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(n3x3_reduce-mask_nums[1]),
            nn.ReLU(True),
            nn.Conv2d(n3x3_reduce-mask_nums[1], n3x3-mask_nums[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n3x3-mask_nums[2]),
            nn.ReLU(True),
        )
        
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduce-mask_nums[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(n5x5_reduce-mask_nums[3]),
            nn.ReLU(True),
            nn.Conv2d(n5x5_reduce-mask_nums[3], n5x5_1-mask_nums[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x5_1-mask_nums[4]),
            nn.ReLU(True),
            nn.Conv2d(n5x5_1-mask_nums[4], n5x5_2-mask_nums[5], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x5_2-mask_nums[5]),
            nn.ReLU(True),
        )
        
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels-mask_nums[6], kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_channels-mask_nums[6]),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = self.b4(x)
        return torch.cat([out1,out2,out3,out4], 1)

class GoogLeNet(nn.Module):
    def __init__(self, inception, num_classes=10, mask_nums=None):
        super(GoogLeNet, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * 64
        cnt = 0
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192-self.mask_nums[cnt], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192-self.mask_nums[cnt]),
            nn.ReLU(True),
        )
        
        self.inception_a3 = inception(192-self.mask_nums[cnt], 64, 96, 128, 16, 32, 32, 32, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        self.inception_b3 = inception(256-cat_mask_num, 128, 128, 192, 32, 96, 96, 64, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception_a4 = inception(480-cat_mask_num, 192, 96, 208, 16, 48, 48, 64, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        self.inception_b4 = inception(512-cat_mask_num, 160, 112, 224, 24, 64, 64, 64, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        self.inception_c4 = inception(512-cat_mask_num, 128, 128, 256, 24, 64, 64, 64, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        self.inception_d4 = inception(512-cat_mask_num, 112, 144, 288, 32, 64, 64, 64, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        self.inception_e4 = inception(528-cat_mask_num, 256, 160, 320, 32, 128, 128, 128, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        
        self.inception_a5 = inception(832-cat_mask_num, 256, 160, 320, 32, 128, 128, 128, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        self.inception_b5 = inception(832-cat_mask_num, 384, 192, 384, 48, 128, 128, 128, self.mask_nums[cnt+1:cnt+8])
        cat_mask_num, cnt = self.mask_nums[cnt+1]+self.mask_nums[cnt+3]+self.mask_nums[cnt+6]+self.mask_nums[cnt+7], cnt+7
        
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024-cat_mask_num, num_classes)
    
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.inception_a3(out)
        out = self.inception_b3(out)
        out = self.maxpool(out)
        out = self.inception_a4(out)
        out = self.inception_b4(out)
        out = self.inception_c4(out)
        out = self.inception_d4(out)
        out = self.inception_e4(out)
        out = self.maxpool(out)
        out = self.inception_a5(out)
        out = self.inception_b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MobileNet_V1(nn.Module):
    def __init__(self, num_classes=10, mask_nums=None):
        super(MobileNet_V1, self).__init__()
        
        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        
        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        
        self.mask_nums = mask_nums if mask_nums else [0] * 14
        self.model = nn.Sequential(
            conv_bn(3, 32-self.mask_nums[0], 1),
            conv_dw(32-self.mask_nums[0], 64-self.mask_nums[1], 1),
            conv_dw(64-self.mask_nums[1], 128-self.mask_nums[2], 2),
            conv_dw(128-self.mask_nums[2], 128-self.mask_nums[3], 1),
            conv_dw(128-self.mask_nums[3], 256-self.mask_nums[4], 2),
            conv_dw(256-self.mask_nums[4], 256-self.mask_nums[5], 1),
            conv_dw(256-self.mask_nums[5], 512-self.mask_nums[6], 2),
            conv_dw(512-self.mask_nums[6], 512-self.mask_nums[7], 1),
            conv_dw(512-self.mask_nums[7], 512-self.mask_nums[8], 1),
            conv_dw(512-self.mask_nums[8], 512-self.mask_nums[9], 1),
            conv_dw(512-self.mask_nums[9], 512-self.mask_nums[10], 1),
            conv_dw(512-self.mask_nums[10], 512-self.mask_nums[11], 1),
            conv_dw(512-self.mask_nums[11], 1024-self.mask_nums[12], 2),
            conv_dw(1024-self.mask_nums[12], 1024-self.mask_nums[13], 1),
            nn.AvgPool2d(2))
        self.fc = nn.Linear(1024-self.mask_nums[13], num_classes)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, mask_nums):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        
        channels = expansion * in_channels - mask_nums[1]
        in_channels = in_channels - mask_nums[0]
        out_channels = out_channels - mask_nums[2]
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNet_V2(nn.Module):
    def __init__(self, invertedresidual, num_classes=10, mask_nums=None):
        super(MobileNet_V2, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * 36
        self.inverted_residual_setting = [(1,  16, 1, 1),
                                          (6,  24, 2, 1),
                                          (6,  32, 3, 2),
                                          (6,  64, 4, 2),
                                          (6,  96, 3, 1),
                                          (6, 160, 3, 2),
                                          (6, 320, 1, 1)]
        first_out_channels = 32 - self.mask_nums[0]
        self.conv1 = nn.Conv2d(3, first_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(first_out_channels)
        self.layers = self._make_layers(invertedresidual, in_channels=32)
        last_out_channels = 1280 - self.mask_nums[-1]
        self.conv2 = nn.Conv2d(320-self.mask_nums[-2], last_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(last_out_channels)
        self.linear = nn.Linear(last_out_channels, num_classes)

    def _make_layers(self, invertedresidual, in_channels):
        layers, cnt = [], 1
        for expansion, out_channels, num_blocks, stride in self.inverted_residual_setting:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(invertedresidual(in_channels, out_channels, expansion, stride, self.mask_nums[cnt-1:cnt+2]))
                in_channels = out_channels
                cnt += 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def vgg16_bn(num_classes=10, mask_nums=None):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]
    return VGGNet(cfg=cfg, num_classes=num_classes, mask_nums=mask_nums)

def resnet20(num_classes=10, mask_nums=None):
    return ResNet(ResidualBlock, [3, 3, 3], num_classes=num_classes, mask_nums=mask_nums)

def resnet32(num_classes=10, mask_nums=None):
    return ResNet(ResidualBlock, [5, 5, 5], num_classes=num_classes, mask_nums=mask_nums)

def resnet44(num_classes=10, mask_nums=None):
    return ResNet(ResidualBlock, [7, 7, 7], num_classes=num_classes, mask_nums=mask_nums)

def resnet56(num_classes=10, mask_nums=None):
    return ResNet(ResidualBlock, [9, 9, 9], num_classes=num_classes, mask_nums=mask_nums)

def resnet110(num_classes=10, mask_nums=None):
    return ResNet(ResidualBlock, [18, 18, 18], num_classes=num_classes, mask_nums=mask_nums)

def densenet40(num_classes=10, mask_nums=None):
    return DenseNet(DenseBlock, Transition, 40, num_classes=num_classes, mask_nums=mask_nums)

def googlenet(num_classes=10, mask_nums=None):
    return GoogLeNet(Inception, num_classes=num_classes, mask_nums=mask_nums)

def mobilenet_v1(num_classes=10, mask_nums=None):
    return MobileNet_V1(num_classes=num_classes, mask_nums=mask_nums)

def mobilenet_v2(num_classes=10, mask_nums=None):
    return MobileNet_V2(InvertedResidual, num_classes=num_classes, mask_nums=mask_nums)
