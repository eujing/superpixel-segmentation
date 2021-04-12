from collections import defaultdict
import os
import json

import numpy as np
from PIL import Image
import skimage.io as io
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class COCODataset(Dataset):
    """Class to store COCO semantic segmentation dataset"""

    def __init__(self,
            ann_file_path = "./data/annotations/panoptic_val2017.json",
            class_labels_dir = "./data/annotations/classes_val2017",
            image_dir = "./data/val2017",
            catid_to_classid = None, device = None):

        self.ann_file_path = ann_file_path
        self.class_labels_dir = class_labels_dir
        self.image_dir = image_dir

        with open(ann_file_path, "r") as f:
            self.js = json.load(f)

        # Mappings from category ids to contiguous integers
        if catid_to_classid is None:
            cat_dict = self.get_categories_dict()
            cat_ids = sorted(cat_dict.keys())
            self.catid_to_classid = {
                catid: i
                for catid, i
                in zip(cat_ids, range(len(cat_ids)))}
        else:
            self.catid_to_classid = catid_to_classid
        self.classid_to_catid = {i: catid for catid, i in self.catid_to_classid.items()}

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def __len__(self):
        return len(self.js["annotations"])

    @staticmethod
    def rgb2id(color):
        """this function is copied from panopticapi.utils 
        https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    @staticmethod
    def id2rgb(id_map):
        """this function is copied from panopticapi.utils 
        https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
        """
        if isinstance(id_map, np.ndarray):
            id_map_copy = id_map.copy()
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map_copy % 256
                id_map_copy //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return color

    def __getitem__(self, index):
        img = io.imread(
            os.path.join(
                self.image_dir, self.js["annotations"][index]["file_name"][:-3] + "jpg"
            )
        )

        # If image happens to be single channel, duplicate to 3 channels
        if len(img.shape) == 2:
            img = np.dstack([img, img, img])

        label = np.load(
            os.path.join(
                self.class_labels_dir, self.js["annotations"][index]["file_name"][:-3] + "npy"
            )
        )

        return (TF.to_tensor(img).to(self.device),
                torch.LongTensor(label).to(self.device))

    def get_categories_dict(self):
        temp_dict = {cate["id"]: cate["name"] for cate in self.js["categories"]}
        temp_dict[0] = "unlabeled"
        return temp_dict

    def get_supercategories_dict(self):
        temp_dict = {
            cate["id"]: cate["supercategory"] for cate in self.js["categories"]
        }
        temp_dict[0] = "unlabeled"
        return temp_dict

VOC_CATEGORIES = [
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
    "potted plant", "sheep", "couch", "train", "tv"]

class COCODataset2(Dataset):
    """Class to store COCO semantic segmentation dataset"""

    def __init__(self,
            ann_file_path = "./data/annotations/panoptic_val2017.json",
            ann_segments_dir = "./data/annotations/panoptic_val2017",
            image_dir = "./data/val2017",
            catid_to_classid = None, subset_categories = None,
            device = None):

        self.ann_file_path = ann_file_path
        self.ann_segments_dir = ann_segments_dir
        self.image_dir = image_dir

        with open(ann_file_path, "r") as f:
            self.js = json.load(f)

        # Mappings from category ids to contiguous integers
        self.subset_categories = subset_categories
        if catid_to_classid is None:
            cat_dict = self.get_categories_dict()
            if subset_categories is None:
                cat_ids = sorted(cat_dict.keys())
            else:
                cat_ids = [
                    cat_id
                    for cat_id, cat_name in cat_dict.items()
                    if cat_name in subset_categories]

            # Map all other categories to 0
            self.catid_to_classid = defaultdict(lambda: 0)

            for cat_id, i in zip(cat_ids, range(len(cat_ids))):
                self.catid_to_classid[cat_id] = i + 1

        else:
            self.catid_to_classid = catid_to_classid

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.js["annotations"])

    @staticmethod
    def rgb2id(color):
        """this function is copied from panopticapi.utils 
        https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    @staticmethod
    def id2rgb(id_map):
        """this function is copied from panopticapi.utils 
        https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
        """
        if isinstance(id_map, np.ndarray):
            id_map_copy = id_map.copy()
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map_copy % 256
                id_map_copy //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return color

    def __getitem__(self, index):
        annotation = self.js["annotations"][index]

        img = io.imread(
            os.path.join(
                self.image_dir, annotation["file_name"][:-3] + "jpg"
            )
        )

        # If image happens to be single channel, duplicate to 3 channels
        if len(img.shape) == 2:
            img = np.dstack([img, img, img])

        labels = io.imread(
            os.path.join(
                self.ann_segments_dir, annotation["file_name"]
            )
        )
        labels = COCODataset.rgb2id(labels)
        class_labels = labels.copy()
        for segment in annotation["segments_info"]:
            k, v = segment["id"], segment["category_id"]
            class_labels[labels == k] = self.catid_to_classid[v]

        return (self.preprocess(img).to(self.device),
                torch.LongTensor(class_labels).to(self.device))

    def get_categories_dict(self):
        temp_dict = {cate["id"]: cate["name"] for cate in self.js["categories"]}
        temp_dict[0] = "unlabeled"
        return temp_dict

    def get_supercategories_dict(self):
        temp_dict = {
            cate["id"]: cate["supercategory"] for cate in self.js["categories"]
        }
        temp_dict[0] = "unlabeled"
        return temp_dict

