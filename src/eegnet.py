import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    Compact Convolutional Neural Network for EEG Motor Imagery (Lawhern et al.).
    Input shape: (batch_size, 1, n_channels, n_samples)
    """
    def __init__(self, n_classes=2, channels=21, samples=480, F1=8, D=2, F2=16, kernel_length=64, drop_rate=0.25):
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Conv -> Spatial Depthwise Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(drop_rate)
        
        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(drop_rate)
        
        # Classifier
        out_samples = samples // 4 // 8  # 480 -> 120 -> 15
        self.fc = nn.Linear(F2 * out_samples, n_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Linear Classifier
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x