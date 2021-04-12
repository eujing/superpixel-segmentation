import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco_utils import get_coco
from models import ReprModel
import presets
import utils

def get_transform(train):
    base_size = 520
    crop_size = 480

    if train:
        return presets.SegmentationPresetTrain(base_size, crop_size)
    else:
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

                if (i+1) % 100 == 0:
                    print(f"[Epoch {epoch}, i={i}] Avg. Training loss: {total_loss / 100}")
                    total_loss = 0.0

            except Exception as err:
                print(err)
                import pdb; pdb.set_trace()

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


if __name__ == "__main__":
    NUM_FEATURES = 100
    NUM_CLASSES = 21

    device = torch.device("cuda")

    train_dataset = get_coco("./data", "train", get_transform(train=True))
    train_batches = DataLoader(
            train_dataset, batch_size=4, num_workers=4,
            collate_fn=utils.collate_fn, shuffle=True)

    val_dataset = get_coco("./data", "val", get_transform(train=False))
    val_batches = DataLoader(val_dataset, batch_size=1, collate_fn=utils.collate_fn)

    with open("catid_to_classid.pkl", "rb") as f:
        catid_to_classid = pickle.load(f)

    # Load the pretrained FCN, mainly for its backbone
    cnn_pretrained = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True)
    # Load up a new FCN head on top of the backbone to learn feature vectors instead
    cnn_repr = ReprModel(cnn_pretrained.backbone, NUM_FEATURES, NUM_CLASSES).to(device)

    # Finetune the CNN to produce pixelwise feature vectors
    finetune_cnn(cnn_repr, train_batches, 1, device, lr=0.005)
    torch.save(cnn_repr.state_dict(), "cnn_weights_hinge.pt")

    # Evaluate segmentation metrics on the validation set
    confmat = evaluate_cnn(cnn_repr, val_batches, NUM_CLASSES, device)
    print(confmat)
