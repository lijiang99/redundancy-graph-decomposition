import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, cfg, num_classes=1000, dropout=0.5, mask_nums=None):
        super(VGGNet, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * len(cfg)
        self.features = self._make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear((512-self.mask_nums[-1])*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
    
    def _make_layers(self, cfg):
        layers = []
        in_channels, cnt = 3, 0
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = x - self.mask_nums[cnt]
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                in_channels, cnt = out_channels, cnt + 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, mask_num1, mask_num2, downsample_mask_num,
                 stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(in_channels, width-mask_num1)
        self.bn1 = nn.BatchNorm2d(width-mask_num1)
        self.conv2 = conv3x3(width-mask_num1, width-mask_num2, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width-mask_num2)
        self.conv3 = conv1x1(width-mask_num2, out_channels*self.expansion-downsample_mask_num)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion-downsample_mask_num)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, mask_nums=None):
        super(ResNet, self).__init__()
        self.mask_nums = mask_nums if mask_nums else [0] * (sum(layers) * 3 + len(layers) + 1)
        self.cnt = 0
        self.in_channels = 64 - self.mask_nums[0]
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cnt += 1
        self.layer1 = self._make_layer(block, 64, layers[0], first_layer=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion-self.mask_nums[-(2*layers[3]-1)], num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1, first_layer=False):
        downsample, downsample_mask_num = None, self.mask_nums[self.cnt+2]
        downsample_out_channels = out_channels * block.expansion - downsample_mask_num
        previous_dilation = self.dilation
        if stride != 1 or first_layer:
            downsample = nn.Sequential(conv1x1(self.in_channels, downsample_out_channels, stride),
                                       nn.BatchNorm2d(downsample_out_channels))
        
        layers = []
        mask1, mask2 = tuple(self.mask_nums[self.cnt:self.cnt+2])
        layers.append(block(self.in_channels, out_channels, mask1, mask2, downsample_mask_num,
                            stride, downsample, self.groups, self.base_width, previous_dilation))
        self.in_channels = downsample_out_channels
        self.cnt += 3
        for i in range(1, blocks):
            mask1, mask2 = tuple(self.mask_nums[self.cnt:self.cnt+2])
            layers.append(block(self.in_channels, out_channels, mask1, mask2, downsample_mask_num,
                                groups=self.groups, base_width=self.base_width, dilation=self.dilation))
            self.cnt += 2
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def vgg16_bn(num_classes=1000, mask_nums=None):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    return VGGNet(cfg=cfg, num_classes=num_classes, mask_nums=mask_nums)

def vgg19_bn(num_classes=1000, mask_nums=None):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    return VGGNet(cfg=cfg, num_classes=num_classes, mask_nums=mask_nums)

def resnet50(num_classes=1000, mask_nums=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, mask_nums=mask_nums)