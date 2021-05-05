import random
import pdb
import traceback

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import coco_utils
import models
import presets
import utils


def finetune_cnn(model, train_batches, num_epochs, device, lr=0.01):
    """
    This function runs a training loop for the FCN semantic segmentation model
    """
    print("Training CNN...")

    # Set train mode since we have batch norm layers
    model.train()

    structured = isinstance(model, models.StructSVMModel)
    if structured:
        criterion = utils.struct_hinge_criterion
        desc = "Train S-SVM"
    else:
        criterion = utils.hinge_criterion
        desc = "Train L-SVM"

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0

        progress = tqdm(train_batches, dynamic_ncols=True, desc=desc)

        for i, batch in enumerate(progress):
            try:
                optimizer.zero_grad()

                images, pxl_labels, superpixels, edges = batch
                images, pxl_labels = images.to(device), pxl_labels.to(device)
                superpixels, edges = superpixels.to(device), edges.to(device)

                sp_labels = utils.pixel_labels_to_superpixel(pxl_labels, superpixels)

                if structured:
                    node_potentials, edge_potentials = model(images, superpixels, edges)
                    loss = criterion(
                            model,
                            node_potentials, edge_potentials,
                            sp_labels, edges,
                            ignore_index=255)

                else:
                    node_potentials = model(images, superpixels)
                    loss = criterion(node_potentials, sp_labels, ignore_index=255)

                assert not loss.isnan().any()

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    progress.write(
                        f"[Epoch {epoch}, i={i}] Avg. Training loss: {total_loss / 100}"
                    )
                    total_loss = 0.0

            except Exception:
                traceback.print_exc()
                pdb.set_trace()


def evaluate_cnn(model, test_batches, num_classes, device):
    # Track metrics
    confmat = utils.ConfusionMatrix(num_classes)

    model.eval()

    structured = isinstance(model, models.StructSVMModel)
    if structured:
        desc = "Eval S-SVM"
    else:
        desc = "Eval L-SVM"

    with torch.no_grad():
        progress = tqdm(test_batches, dynamic_ncols=True, desc=desc)

        for i, batch in enumerate(progress):
            images, pxl_labels, superpixels, edges = batch
            images, pxl_labels = images.to(device), pxl_labels.to(device)
            superpixels, edges = superpixels.to(device), edges.to(device)

            if structured:
                node_potentials, edge_potentials = model(images, superpixels, edges)
                x_pred, _ = model.infer(node_potentials, edge_potentials, edges)
                pred_sp_labels = x_pred.argmax(dim=1)

            else:
                node_potentials = model(images, superpixels)
                pred_sp_labels = node_potentials.argmax(dim=1)

            pred_pxl_labels = utils.superpixel_labels_to_pixel(pred_sp_labels, superpixels)

            confmat.update(pxl_labels.flatten(), pred_pxl_labels.flatten())

    return confmat


if __name__ == "__main__":
    NUM_FEATURES = 100
    NUM_CLASSES = 21
    superpixel_kwargs = {
        "n_segments": 200,
        "compactness": 50.0,
        "max_iter": 5,
        "sigma": 0,
        "multichannel": True,
        "convert2lab": True,
        "enforce_connectivity": False,
        "min_size_factor": 0.5,
        "max_size_factor": 3,
        "start_label": 0,
    }
    device = torch.device("cuda")
    SEED = 42
    SUBSET_SIZE = 5000

    # Set all random seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_dataset_sp = coco_utils.get_coco_superpixel(
        image_dir="./data/train2017/",  # TODO:change this to data directory which contains images
        annFile_path="./data/annotations/instances_train2017.json",  # TODO: change this to the path to json file of corresponding data
        image_set="train",
        transforms=presets.SegmentationPresetEval(520),
        # superpixel_dir="./data/sp_train",
        superpixel_kwargs=superpixel_kwargs,
        normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    val_dataset_sp = coco_utils.get_coco_superpixel(
        image_dir="./data/val2017/", # TODO:change this to data directory which contains images
        annFile_path="./data/annotations/instances_val2017.json", # TODO: change this to the path to json file of corresponding data
        image_set="val",
        transforms=presets.SegmentationPresetEval(520),
        # superpixel_dir="./data/sp_val",
        superpixel_kwargs=superpixel_kwargs,
        normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )

    # Convert the whole dataset and saved superpixel results in local
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

    train_indices = torch.randperm(len(train_dataset_sp))[:SUBSET_SIZE]
    train_batches = DataLoader(
        train_dataset_sp,
        batch_size=1,
        num_workers=1,
        # collate_fn=utils.collate_fn,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    val_indices = torch.randperm(len(val_dataset_sp))
    val_batches = DataLoader(
        val_dataset_sp,
        batch_size=1,
        num_workers=1,
        # collate_fn=utils.collate_fn,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )

    cnn_pretrained = torch.hub.load(
        "pytorch/vision:v0.9.0", "fcn_resnet101", pretrained=True
    )
    # Load up a new FCN head on top of the backbone to learn feature vectors instead
    cnn_repr = (
        models.ReprModel(cnn_pretrained.backbone, NUM_FEATURES, NUM_CLASSES).to(device).eval()
    )

    # Select model to train
    svm_model = models.BaselineSVMModel(cnn_repr.get_repr, NUM_FEATURES, NUM_CLASSES).to(device)
    # svm_model = models.StructSVMModel(cnn_repr.get_repr, NUM_FEATURES, NUM_CLASSES).to(device)

    # Finetune the SVM to produce superpixel-wise feature vectors
    finetune_cnn(svm_model, train_batches, 1, device, lr=0.005)

    torch.save(svm_model.state_dict(), "lsvm_sp_weights.pt")
    # torch.save(svm_model.state_dict(), "ssvm_sp_weights.pt")

    # Evaluate pixel-wise segmentation metrics on the validation set
    conf_mat = evaluate_cnn(svm_model, val_batches, NUM_CLASSES, device)
    print(conf_mat)
