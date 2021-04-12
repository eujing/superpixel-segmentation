import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # out: Batch x Width x Height x Classes
        out = self.linear(repr)

        # Swap back to Batch x Classes x Width x Height
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

