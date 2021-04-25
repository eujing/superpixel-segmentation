
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import coco_utils
from coco_utils import get_coco
from models import ReprModel
import presets

import utils

def get_transform(train):
    base_size = 520
    # crop_size = 480
    return presets.SegmentationPresetEval(base_size)


def hinge_criterion(output, labels, margin=1, ignore_index=-1):
    # output: B x C x W x H
    # labels: B x W x H

    mask = labels != ignore_index

    # For pixels to ignore, temporarily use 0 as the label index
    labels_index = torch.where(labels == ignore_index, 0, labels)
    x_correct = torch.gather(output, dim=1, index=labels_index.unsqueeze(dim=1))

    # Average across classes, loss: B x W x H
    loss = torch.mean(F.relu(margin - (x_correct - output)), dim=1)

    # Average over all batches and pixels, for masked regions only
    return torch.sum(mask.float() * loss) / torch.sum(mask)


def finetune_cnn(model, train_batches, num_epochs, device, lr=0.01):
    """
    This function runs a training loop for the FCN semantic segmentation model
    """
    print("Training CNN...")

    # Set train mode since we have batch norm layers
    model.train()

    # But make sure backbone stays in eval mode
    model.backbone.eval()

    # TODO: Shall we do class weights?
    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 4]))
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, batch in enumerate(tqdm(train_batches)):
            try:
                optimizer.zero_grad()

                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                output = model(images)

                # loss = criterion(output, labels)
                loss = hinge_criterion(output, labels, ignore_index=255)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(
                        f"[Epoch {epoch}, i={i}] Avg. Training loss: {total_loss / 100}"
                    )
                    total_loss = 0.0

            except Exception as err:
                print(err)
                import pdb

                pdb.set_trace()


def evaluate_cnn(model, test_batches, num_classes, device):
    # Track metrics
    confmat = utils.ConfusionMatrix(num_classes)

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_batches)):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            output = model(images)[0]
            preds = output.argmax(0)

            confmat.update(labels.flatten(), preds.flatten())

    return confmat

class SvmModel(nn.Module):
    def __init__(self, backbone, n_feat, n_classes):
        super().__init__()
        self.backbone = backbone
        self.n_feat_backbone = n_feat
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.nn_node = nn.Linear(n_feat, n_classes, bias=False)
        self.nn_edge = nn.Linear(n_feat * 2, n_classes, bias=False)

    def _node_feature_aggregation(self, features, superpixel):
        """
        features: torch.tensor(1, H, W, n_feat_backbone)
        superpixel torch.LongTensor(H, W), value:superpixel label
        return: node_features: torch.tensor(num_superpixels, n_feat)
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

    def _edge_feature_aggregation(self, node_features, edge_indexes):
        """
        node_features: torch.tensor(num_superpixels, n_feat)
        edge_indexes: torch.LongTensor(num_edges, 2), value:node indexes of the edge
        return: edge_features: torch.tensor(num_edges, 2*n_feat)
        """
        nodes_in_edges = node_features[edge_indexes, :]
        edge_features = nodes_in_edges.reshape((-1, 2 * self.n_feat))
        return edge_features

    def forward(self, image, superpixel, edge_indexes):
        """
        image: torch.tensor(1, C, H, W)
        superpixel torch.LongTensor(H, W), value:superpixel label
        edge_indexes: torch.LongTensor(num_edges, 2), value:node indexes of the edge
        return:
        node_potentials: torch.tensor(num_nodes, n_classes)
        edge_potentials: torch.tensor(num_edges, n_classes)
        """
        with torch.no_grad():
            features = self.backbone(image)
        node_features = self._node_feature_aggregation(features, superpixel)
        edge_features = self._edge_feature_aggregation(node_features, edge_indexes)
        node_potentials = self.nn_node(node_features)
        edge_potentials = self.nn_edge(edge_features)
        return node_potentials, edge_potentials

if __name__=='__main__':
    NUM_FEATURES = 100
    NUM_CLASSES = 21
    device = torch.device("cpu")

    train_dataset_sp = coco_utils.get_coco_superpixel(
        image_dir="./data/train2017/", #TODO:change this to data directory which contains images
        annFile_path="./data/annotations/instances_train2017.json", #TODO: change this to the path to json file of corresponding data
        image_set="train",
        transforms=presets.SegmentationPresetEval(520),
        # superpixel_dir="./data/sp_train",
    )
    #
    # # Convert the whole dataset and saved superpixel results in local
    # if isinstance(train_dataset_sp, torch.utils.data.Subset):
    #     train_dataset_sp.dataset.convert_dataset(
    #         parallel=True, processes=4,
    #         # start_index=0, end_index=None
    #     )
    # else:
    #     train_dataset_sp.convert_dataset(
    #         parallel=True, processes=4,
    #         # start_index=0, end_index=None
    #     )

    train_batches = DataLoader(
        train_dataset_sp,
        batch_size=4,
        num_workers=2,
        # collate_fn=utils.collate_fn,
        shuffle=False,
    )
    cnn_pretrained = torch.hub.load(
        "pytorch/vision:v0.9.0", "fcn_resnet101", pretrained=True
    )
    # Load up a new FCN head on top of the backbone to learn feature vectors instead
    cnn_repr = (
        ReprModel(cnn_pretrained.backbone, NUM_FEATURES, NUM_CLASSES).to(device).eval()
    )
    svm_model = SvmModel(cnn_repr.get_repr, NUM_FEATURES, NUM_CLASSES).to(device)

    img, labels, sp, edges = train_dataset_sp[0]
    img = torch.unsqueeze(img, 0)
    sp = torch.LongTensor(sp.astype(int))
    edges = torch.LongTensor(edges.astype(int))
    node_potentials, edge_potentials = svm_model(img, sp, edges)

    x, y = utils.integer_linear_programming(
        node_potentials.detach().to('cpu').numpy(), edges.detach().to('cpu').numpy(), edge_potentials.detach().to('cpu').numpy()
    )


