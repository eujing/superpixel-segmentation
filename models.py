import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """Class defining AlexNet layers used for the convolutional network"""

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class FCNHead(nn.Sequential):
    """Class defining FCN (fully convolutional network) layers"""

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)

class ReprModel(nn.Module):
    def __init__(self, backbone, n_feat, n_classes):
        super(ReprModel, self).__init__()
        self.n_feat = n_feat
        self.backbone = backbone
        self.classifier = FCNHead(2048, n_feat)
        self.linear = nn.Linear(n_feat, n_classes)

        self.backbone.eval()

    def forward(self, x):
        repr = self.get_repr(x)
        out = self.linear(repr)

        # Swap back to Batch x Width x Height x Features
        # to be compatible with nn.CrossEntropyLoss
        out = out.permute(0, 3, 1, 2)
        return out

    def get_repr(self, x):
        input_shape = x.shape[-2:]

        # Assume backbone is pre-trained and frozen
        with torch.no_grad():
            features = self.backbone(x)["out"]

        # Train the classifier head only
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        # Batch x Width x Height x Features
        x = x.permute(0, 2, 3, 1)

        return x

