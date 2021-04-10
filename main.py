import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import COCODataset
from models import ReprModel

def pretrain_cnn(model, train_batches, num_epochs, lr=0.0005):
    """
    This function runs a training loop for the FCN semantic segmentation model
    """
    N = len(train_batches)
    print("Training CNN...")

    # Set train mode since we have batch norm layers
    model.train()

    # TODO: Shall we do class weights?
    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 4]))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()

            images, labels = batch
            output = model(images)
            labels = labels.contiguous().view(-1, 1).squeeze()

            loss = criterion(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch}] Training loss: {total_loss / N}")


if __name__ == "__main__":
    NUM_FEATURES = 100

    device = torch.device("cuda")

    with open("catid_to_classid.pkl", "rb") as f:
        catid_to_classid = pickle.load(f)

    train_dataset = COCODataset(
            ann_file_path="./data/annotations/panoptic_train2017.json",
            class_labels_dir="./data/annotations/classes_train2017",
            image_dir="./data/train2017",
            catid_to_classid=catid_to_classid,
            device=device)
    train_batches = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = COCODataset(
            ann_file_path="./data/annotations/panoptic_val2017.json",
            class_labels_dir="./data/annotations/classes_val2017",
            image_dir="./data/val2017",
            catid_to_classid=catid_to_classid,
            device=device)
    val_batches = DataLoader(val_dataset, batch_size=1, shuffle=True)

    num_classes = len(train_dataset.catid_to_classid)
    cnn = ReprModel(NUM_FEATURES, num_classes).to(device)
    pretrain_cnn(cnn, train_batches, 1)
    torch.save(cnn.state_dict(), "cnn_weights.pt")
