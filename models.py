import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

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
        """
        Args:
            x : torch.tensor(B, C, H, W)
        Returns:
            features: torch.tensor(B, H, W, n_feat)
        """
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

class BaselineSVMModel(nn.Module):
    def __init__(self, backbone, n_feat, n_classes):
        super().__init__()
        self.backbone = backbone
        self.n_feat_backbone = n_feat
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.nn_node = nn.Linear(n_feat, n_classes, bias=False)

    def _node_feature_aggregation(self, features, superpixel):
        """
        Args:
            features: torch.tensor(1, H, W, n_feat_backbone)
            superpixel torch.LongTensor(H, W), value: superpixel label
        Returns:
            node_features: torch.tensor(num_superpixels, n_feat)
        """
        sp_flat = superpixel.reshape(-1)
        features_flat = features.reshape((-1, self.n_feat_backbone))
        # Calculate mean feature for each superpixel cluster
        num_sp = torch.max(sp_flat)
        M = torch.zeros(num_sp + 1, len(features_flat)).to(features.device)
        M[sp_flat, torch.arange(len(features_flat))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        node_features = torch.mm(M, features_flat)
        return node_features

    def forward(self, image, superpixel):
        """
        Args:
            image: torch.tensor(1, C, H, W)
            superpixel torch.LongTensor(1, H, W), value: superpixel label
        Returns:
            node_potentials: torch.tensor(num_nodes, num_classes)
        """
        with torch.no_grad():
            features = self.backbone(image)
        node_features = self._node_feature_aggregation(features, superpixel)
        node_potentials = self.nn_node(node_features)
        return node_potentials

class StructSVMModel(BaselineSVMModel):
    def __init__(self, backbone, n_feat, n_classes):
        super().__init__(backbone, n_feat, n_classes)
        self.nn_edge = nn.Linear(n_feat * 2, n_classes, bias=False)

    def _edge_feature_aggregation(self, node_features, edge_indexes):
        """
        Args:
            node_features: torch.tensor(num_superpixels, n_feat)
            edge_indexes: torch.LongTensor(num_edges, 2), value: node indexes of the edge
        Returns:
            edge_features: torch.tensor(num_edges, 2*n_feat)
        """
        nodes_in_edges = node_features[edge_indexes, :]
        edge_features = nodes_in_edges.reshape((-1, 2 * self.n_feat))
        return edge_features

    def forward(self, image, superpixel, edge_indexes):
        """
        Args:
            image: torch.tensor(1, C, H, W)
            superpixel torch.LongTensor(1, H, W), value: superpixel label
            edge_indexes: torch.LongTensor(num_edges, 2), value: node indexes of the edge
        Returns:
            node_potentials: torch.tensor(num_nodes, num_classes)
            edge_potentials: torch.tensor(num_edges, num_classes)
        """
        with torch.no_grad():
            features = self.backbone(image)
        node_features = self._node_feature_aggregation(features, superpixel)
        edge_features = self._edge_feature_aggregation(node_features, edge_indexes)
        node_potentials = self.nn_node(node_features)
        edge_potentials = self.nn_edge(edge_features)
        return node_potentials, edge_potentials

    def infer(self, node_potentials, edge_potentials, edge_indexes, x_true=None):
        """
        Args:
            node_potentials: torch.tensor(num_nodes, num_classes)
            edge_potentials: torch.tensor(num_edges, num_classes)
            edge_indexes: torch.LongTensor(1, num_edges, 2), value: node indexes of the edge
            x_true: torch.LongTensor(num_nodes, num_classes), value: binary
        Returns:
            x: torch.LongTensor(num_nodes, num_classes), value: binary
            y: torch.LongTensor(num_edges, num_classes), value: binary
        """
        if x_true is not None:
            x_true = x_true.detach().to("cpu").numpy()

        x, y = utils.integer_linear_programming(
            node_potentials.detach().to("cpu").numpy(),
            edge_indexes.squeeze(dim=0).detach().to("cpu").numpy(),
            edge_potentials.detach().to("cpu").numpy(),
            x_true=x_true
        )
        x = torch.LongTensor(x).to(node_potentials.device)
        y = torch.LongTensor(y).to(edge_potentials.device)

        return x, y
