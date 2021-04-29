import torch
import numpy as np
import itertools

from ortools.linear_solver import pywraplp


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


def integer_linear_programming(node_potentials, edge_pair_indexes, edge_potentials):
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

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # set up vars for lp solver
    node_vars = [[solver.IntVar(0, 1, f'n{i}_{j}') for i in range(num_classes)] for j in range(num_nodes)]
    edge_vars = [[solver.IntVar(0, 1, f'n{i}_{j}') for i in range(num_classes)] for j in range(num_edges)]
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
        objective.SetCoefficient(node_vars[i][c], float(node_potentials[i, c]))
    for i, c in itertools.product(range(num_edges), range(num_classes)):
        objective.SetCoefficient(edge_vars[i][c], float(edge_potentials[i, c]))
    objective.SetMaximization()
    
    solver.Solve()
    x = np.array([[node_vars[i][c].solution_value() for c in range(num_classes)] for i in range(num_nodes)])
    y = np.array([[edge_vars[i][c].solution_value() for c in range(num_classes)] for i in range(num_edges)])
    return x, y
