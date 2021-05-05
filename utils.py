import torch
import torch.nn.functional as F
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import hashlib

from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver("SCIP")
# solver = pywraplp.Solver("MAPSolver", pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def hinge_criterion(output, labels, margin=1, ignore_index=255):
    # output: ... x n_classes
    # labels: ...

    mask = labels != ignore_index

    # For pixels to ignore, temporarily use 0 as the label index
    labels_index = torch.where(labels == ignore_index, 0, labels)
    # x_correct: ... x 1
    x_correct = torch.gather(output, dim=-1, index=labels_index.unsqueeze(dim=-1))

    # Average across classes
    loss = torch.mean(F.relu(margin - (x_correct - output)), dim=-1)

    # Average over all other dimensions ..., for masked regions only
    return torch.sum(mask.float() * loss) / torch.sum(mask)


def struct_hinge_criterion(
        model,
        node_potentials, edge_potentials,
        sp_labels, edge_indices,
        ignore_index=255):
    """
    Args:
        model: models.StructSVMModel
        node_potentials: torch.tensor(num_nodes, n_classes)
        edge_potentials: torch.tensor(num_edges, n_classes)
        sp_labels: torch.LongTensor(num_nodes), values are class labels
        edge_indices: torch.LongTensor(1, num_edges, 2), values are node indices
    """

    _, n_classes = node_potentials.shape

    mask = sp_labels != ignore_index
    # For superpixels to ignore, temporarily use 0 as the class index
    sp_labels_index = torch.where(sp_labels == ignore_index, 0, sp_labels)

    # True assignments
    x_true = F.one_hot(sp_labels_index, num_classes=n_classes)
    x_true_pairs = x_true[edge_indices.squeeze(dim=0)]
    y_true = x_true_pairs[:, 0, :] * x_true_pairs[:, 1, :]

    # MAP assignments
    x_pred, y_pred = model.infer(node_potentials, edge_potentials, edge_indices, x_true=x_true)

    # Margin is normalized hamming loss over valid superpixels only
    norm_hamming_loss = (x_pred[mask] != x_true[mask]).float().mean()

    # Scoring function over valid nodes and edges only
    mask_x = mask.float().unsqueeze(dim=1)
    mask_x_pairs = mask_x[edge_indices.squeeze(dim=0)]
    mask_y = mask_x_pairs[:, 0, :] * mask_x_pairs[:, 1, :]
    def score(x, y):
        return torch.sum(mask_x * node_potentials * x) + torch.sum(mask_y * edge_potentials * y)

    hinge_loss = F.relu(norm_hamming_loss - (score(x_true, y_true) - score(x_pred, y_pred)))

    return hinge_loss


def integer_linear_programming(
        node_potentials,
        edge_pair_indexes,
        edge_potentials,
        x_true=None):
    """
    All the potentails are acutally log-potentials
    node_potentials.shape=(num_nodes, num_classes)
    edge_pair_indexes.shape=(num_edges, 2)
    edge_pair_indexes[i]=(node_index1, node_index2)
    edge_potentials.shape=(num_edges, num_classes)
    
    log potential for i th node is node_potentials[i]
    
    You can assume all variables are ndarray, and data type in edge_pair_indexes is int
    
    Output: x:(num_nodes, num_classes), y:(num_edges, num_classes). The definitions of x and y are the same as HW3
    """
    num_nodes = node_potentials.shape[0]
    num_edges = edge_potentials.shape[0]
    num_classes = node_potentials.shape[1]

    solver.Clear()

    # set up vars for lp solver
    node_vars = [
        [solver.BoolVar(f"x{i}_{j}") for i in range(num_classes)]
        for j in range(num_nodes)
    ]
    edge_vars = [
        [solver.BoolVar(f"y{i}_{j}") for i in range(num_classes)]
        for j in range(num_edges)
    ]
    # add constraints
    for i in range(num_nodes):
        solver.Add(sum(node_vars[i]) == 1)
    for i, (idx1, idx2) in enumerate(edge_pair_indexes):
        for c in range(num_classes):
            solver.Add(edge_vars[i][c] <= node_vars[idx1][c])
            solver.Add(edge_vars[i][c] <= node_vars[idx2][c])
            solver.Add(edge_vars[i][c] >= node_vars[idx1][c] + node_vars[idx2][c] - 1)

    # define objective
    objective = solver.Objective()
    for i, c in itertools.product(range(num_nodes), range(num_classes)):
        coeff = float(node_potentials[i, c])
        # For Loss Augmented Inference
        if x_true is not None:
            coeff += (1 - 2*float(x_true[i, c])) / (num_nodes * num_classes)
        objective.SetCoefficient(node_vars[i][c], coeff)
    for i, c in itertools.product(range(num_edges), range(num_classes)):
        objective.SetCoefficient(edge_vars[i][c], float(edge_potentials[i, c]))
    objective.SetMaximization()

    solver.Solve()
    x = np.array(
        [
            [node_vars[i][c].solution_value() for c in range(num_classes)]
            for i in range(num_nodes)
        ]
    )
    y = np.array(
        [
            [edge_vars[i][c].solution_value() for c in range(num_classes)]
            for i in range(num_edges)
        ]
    )
    return x, y


def plot_result(
    img, label, predict, model, index, save_dir="./results", show_plot=False
):
    img = img.permute([1, 2, 0]).detach().to("cpu").numpy()
    label = label.detach().to("cpu").numpy()
    predict_label = (
        torch.squeeze(predict).permute([1, 2, 0]).argmax(2).detach().to("cpu").numpy()
    )

    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    img_denormaliz = (
        img * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    )
    img_denormaliz = np.clip(img_denormaliz, a_min=0.0, a_max=1.0)

    vmin = np.min([predict_label.min(), label.min().item()])
    vmax = np.max([predict_label.max(), label.max().item()])
    plt.imshow(img_denormaliz)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(os.path.join(save_dir, f"{model}_{index}_image.png"), dpi=300)
    if show_plot:
        plt.show()
    plt.imshow(predict_label + 1, vmin=vmin + 1, vmax=vmax + 1, norm=colors.LogNorm())
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(os.path.join(save_dir, f"{model}_{index}_predict.png"), dpi=300)

    if show_plot:
        plt.show()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(label + 1, vmin=vmin + 1, vmax=vmax + 1, norm=colors.LogNorm())
    plt.savefig(os.path.join(save_dir, f"{model}_{index}_label.png"), dpi=300)
    if show_plot:
        plt.show()


def encode_paras2name(prefix="", hash_length=40, hash_type="sha3_224", **kwargs):
    sorted_keys = sorted(kwargs.keys())
    name_str = "".join([f"{i}{kwargs[i]}" for i in sorted_keys])
    f_hash = hashlib.new(hash_type)
    f_hash.update(name_str.encode("utf-8"))
    hash_code = f_hash.hexdigest()[-hash_length:]
    return prefix + hash_code


def pixel_labels_to_superpixel(pxl_labels, superpixels):
    """
    pxl_labels: torch.LongTensor(W, H), values are class labels
    superpixels: torch.LongTensor(W, H), values are superpixel assignments

    returns: 
        sp_labels: torch.LongTensor(num_nodes), values are class labels
    """
    num_superpixels = superpixels.max() + 1
    sp_labels = torch.zeros(
            num_superpixels,
            dtype=torch.long,
            device=pxl_labels.device)

    # Majority vote
    for i in range(num_superpixels):
        label_counts = pxl_labels[superpixels == i].bincount()

        if len(label_counts) > 0:
            sp_labels[i] = label_counts.argmax()

    return sp_labels

def superpixel_labels_to_pixel(sp_labels, superpixels):
    """
    sp_labels: torch.LongTensor(num_nodes), values are class labels
    superpixels: torch.LongTensor(W, H), values are superpixel assignments

    returns:
        pxl_labels: torch.LongTensor(W, H), values are class labels
    """

    return sp_labels[superpixels]
