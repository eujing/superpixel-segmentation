import os
import json

import numpy as np
import skimage.io as io
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

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

