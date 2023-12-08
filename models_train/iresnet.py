import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.5):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.inorm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv(x)
        out = self.inorm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.basic_block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, dropout_rate)
        self.basic_block2 = BasicBlock(out_channels, out_channels, kernel_size, stride, padding, dropout_rate)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels)
        ) if stride != 1 else None

    def forward(self, x):
        identity = x

        out = self.basic_block1(x)
        out = self.basic_block2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out

class iResNet(nn.Module):
    def __init__(self, input_channels, num_residual_blocks, out_channels=64, num_classes=11, regression=False):
        super(iResNet, self).__init__()
        self.initial_block = BasicBlock(input_channels, out_channels, kernel_size=1, stride=1, padding=0, dropout_rate=0)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_channels, out_channels) for _ in range(num_residual_blocks)]
        )
        self.ln = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # num classes to num channels
        self.fc1 = nn.Linear(out_channels, num_classes)
        
        self.regression = regression
        if regression:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_block(x)
        x = self.residual_blocks(x)
        x = self.ln(x)
        x = self.act(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        if self.regression:
            x = self.sigmoid(x)
        
        return x

"""
# The input channels would be the sum of:
 features from the attention map (# of heads?), [L,L,N]
 SPOT-1D-Single, [L,L,46]
 and one-hot encoding [L,L,40]

attention_map_channels = N  # This will be the number of attention heads or another dimensionality from the ESM-1b model
spot_1d_channels = 46
one_hot_encoding_channels = 40
input_channels = attention_map_channels + spot_1d_channels + one_hot_encoding_channels

# Instantiate the iResNet model
num_residual_blocks = 12  # Number of residual blocks
num_classes = 11  # Number of classes for classification or regression output
regression = False  # Set to True if the model is for regression

# how the model should be called
network = iResNet(input_channels, num_residual_blocks, num_classes=num_classes, regression=regression)
"""
