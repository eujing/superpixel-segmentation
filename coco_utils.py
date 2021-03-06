# From https://github.com/pytorch/vision/blob/master/references/segmentation/coco_utils.py

import copy
import os
from typing import *

import utils
import json
import copy

import skimage.segmentation as segmentation

import numpy as np

from tqdm import tqdm


from PIL import Image
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from transforms import Compose

from multiprocessing import Pool


class CocoSuperpixel(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        superpixel_dir=None,
        superpixel_kwargs=None,
        normalization=None,
    ):
        """
        root is the dir of image data
        annFile is the path of annotation json file (include the file name)
        superpixel_dir is the dir of superpixel result, if it is None then will calculate superpixel when reading images
        normalization=(mean, std) If any normalization is used in transformation, then this should be set
        """
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.superpixel_kwargs = {
            "n_segments": 100,
            "compactness": 10.0,
            "max_iter": 10,
            "sigma": 0,
            "spacing": None,
            "multichannel": True,
            "convert2lab": None,
            "enforce_connectivity": True,
            "min_size_factor": 0.5,
            "max_size_factor": 3,
            "slic_zero": False,
            "start_label": None,
            "mask": None,
        }
        if superpixel_kwargs is not None:
            self.superpixel_kwargs.update(superpixel_kwargs)
        if normalization is None:
            self.normalization = None
        else:
            self.normalization = np.array(normalization, copy=True)
        self.superpixel_dir = superpixel_dir
        if self.superpixel_dir is not None:
            try:
                with open(os.path.join(self.superpixel_dir, "sp_paras.json"), "r") as f:
                    sp_paras_js = json.load(f)
                if sp_paras_js["hash"] != utils.encode_paras2name(
                    **self.superpixel_kwargs
                ):
                    print(
                        "the paras of superpixel is different with local cached superpixel results!!!"
                    )
            except:
                print("cannot find superpixel paras json file!!!!!")

    def __getitem__(self, index: int):
        """
        return img, target, superpixels, collection_edges
        superpixels: (H, W), ndarray uint16 dtype, value is the label of superpixels clustering (not the label of segmentation).
        collection_edges, (num_edges, 2), ndarray uint16 dtype, each pair is the label of neighboring superpixels.
        """
        img, target = super().__getitem__(index)
        if self.superpixel_dir is None:
            img_sp = self.preprocessing_img_for_sp(img)
            superpixels, collection_edges = CocoSuperpixel.generate_superpixel_results(
                img_sp, self.superpixel_kwargs
            )
        else:
            superpixels, collection_edges = self.read_superpixel_file(index)

        # TODO:convert superpixels to pytorch tensors?
        superpixels = torch.LongTensor(superpixels.astype(int))
        collection_edges = torch.LongTensor(collection_edges.astype(int))

        return img, target, superpixels, collection_edges

    def denormalize_image(self, img):
        if self.normalization is None:
            return img
        mean, std = self.normalization
        img_denormalized = (
            img * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
        )
        img_denormalized = np.clip(img_denormalized, a_min=0.0, a_max=1.0)
        return img_denormalized

    def preprocessing_img_for_sp(self, img):
        img = np.array(img)
        if img.shape[-1] != 3:
            img = np.moveaxis(img, 0, 2)
        img = self.denormalize_image(img)
        return img

    @staticmethod
    def find_superpixels_neighbors(sp):
        sp_center = sp[:-1, :-1, np.newaxis]
        sp_right = sp[:-1, 1:, np.newaxis]
        sp_down = sp[1:, 1:, np.newaxis]
        diff_x = np.concatenate(
            [sp_center + sp_right, np.abs(sp_center - sp_right)], axis=2
        )
        diff_y = np.concatenate(
            [sp_center + sp_down, np.abs(sp_center - sp_down)], axis=2
        )
        collection_diff = np.unique(
            np.append(diff_x[diff_x[..., 1] > 0], diff_y[diff_y[..., 1] > 0], axis=0),
            axis=0,
        )
        collection_edges = np.zeros(collection_diff.shape, dtype="uint16")
        collection_edges[:, 0] = (collection_diff[:, 0] + collection_diff[:, 1]) / 2
        collection_edges[:, 1] = (collection_diff[:, 0] - collection_diff[:, 1]) / 2
        return collection_edges

    @staticmethod
    def generate_superpixel_results(img, sp_kwargs):
        superpixels_labels = segmentation.slic(img, **sp_kwargs)
        uni_labels = np.unique(superpixels_labels)
        for i in range(len(uni_labels)):
            if uni_labels[i] == i:
                continue
            superpixels_labels[superpixels_labels == uni_labels[-1]] = i
            uni_labels = np.insert(uni_labels[:-1], i, i)
        collection_edges = CocoSuperpixel.find_superpixels_neighbors(superpixels_labels)
        return superpixels_labels.astype("uint16"), collection_edges

    @staticmethod
    def generate_sp_write_file(args):
        """
        args=(img, id, superpixel_dir)
        """
        img, id, superpixel_dir, sp_kwargs = args
        (
            superpixels_labels,
            collection_edges,
        ) = CocoSuperpixel.generate_superpixel_results(img, sp_kwargs)
        output_file = os.path.join(superpixel_dir, f"{id}_sp")
        try:
            np.savez(
                output_file, superpixels=superpixels_labels, edges=collection_edges
            )
        except:
            return id

    def convert_dataset(
        self, parallel=True, start_index=0, end_index=None, processes=None
    ):
        """
        Convert the whole dataset and stored in self.superpixel_dir
        TODO: write the paras of superpixel into a txt files
        """
        if self.superpixel_dir is None:
            raise IOError("Please set superpixel_dir")
        if end_index is None:
            end_index = len(self)
        # write sp paras json files
        with open(os.path.join(self.superpixel_dir, "sp_paras.json"), "w") as f:
            sp_paras_js = copy.deepcopy(self.superpixel_kwargs)
            sp_paras_js["hash"] = utils.encode_paras2name(**self.superpixel_kwargs)
            json.dump(sp_paras_js, f)
        if parallel:
            with Pool(processes=processes) as p:
                bad_index = list(
                    tqdm(
                        p.imap(
                            CocoSuperpixel.generate_sp_write_file,
                            (
                                (
                                    self.preprocessing_img_for_sp(
                                        super(CocoSuperpixel, self).__getitem__(i)[0]
                                    ),
                                    self.ids[i],
                                    self.superpixel_dir,
                                    self.superpixel_kwargs,
                                )
                                for i in range(start_index, end_index)
                            ),
                        ),
                        total=(end_index - start_index),
                    )
                )
                p.close()
                p.join()
            bad_index = list(filter(lambda x: x is not None, bad_index))
            print(f"unsuccessful ids: {bad_index}")
        else:
            bad_index = []
            for i in tqdm(range(start_index, end_index)):
                temp = self.generate_sp_write_file(
                    (
                        self.preprocessing_img_for_sp(super().__getitem__(i)[0]),
                        self.ids[i],
                        self.superpixel_dir,
                        self.superpixel_kwargs,
                    )
                )
                if temp is not None:
                    bad_index.append(temp)
            print(f"unsuccessful ids: {bad_index}")

    def read_superpixel_file(self, index):
        id = self.ids[index]
        try:
            file = np.load(os.path.join(self.superpixel_dir, f"{id}_sp.npz"))
            return file["superpixels"], file["edges"]
        except:
            raise IOError(
                f"Cannot read {id}_sp.npz, need to generate superpixels file first"
            )


class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def get_coco(root, image_set, transforms):
    PATHS = {
        "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    CAT_LIST = [
        0,
        5,
        2,
        16,
        9,
        44,
        6,
        3,
        17,
        62,
        21,
        67,
        18,
        19,
        4,
        1,
        64,
        20,
        63,
        7,
        72,
    ]

    transforms = Compose(
        [
            FilterAndRemapCocoCategories(CAT_LIST, remap=True),
            ConvertCocoPolysToMask(),
            transforms,
        ]
    )

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(
        img_folder, ann_file, transforms=transforms
    )

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


def get_coco_superpixel(
    image_dir: str,
    annFile_path: str,
    image_set,
    transforms,
    superpixel_dir=None,
    superpixel_kwargs=None,
    normalization=None,
):
    """
    image_dir is the dir of image data
    annFile_path is the path of annotation json file (include the file name)
    superpixel_dir is the dir of superpixel result, if it is None then will calculate superpixel when reading images
    normalization=(mean, std) If any normalization is used in transformation, then this should be set
    """
    CAT_LIST = [
        0,
        5,
        2,
        16,
        9,
        44,
        6,
        3,
        17,
        62,
        21,
        67,
        18,
        19,
        4,
        1,
        64,
        20,
        63,
        7,
        72,
    ]

    transforms = Compose(
        [
            FilterAndRemapCocoCategories(CAT_LIST, remap=True),
            ConvertCocoPolysToMask(),
            transforms,
        ]
    )
    dataset = CocoSuperpixel(
        image_dir,
        annFile_path,
        transforms=transforms,
        superpixel_dir=superpixel_dir,
        superpixel_kwargs=superpixel_kwargs,
        normalization=normalization,
    )

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)
    return dataset
