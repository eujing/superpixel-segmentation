import os
import json
import pickle
import multiprocessing as mp
import multiprocessing.pool as mpp

import numpy as np
import skimage.io as io
from tqdm import tqdm

from datasets import COCODataset

ANN_SEGMENTS_DIR = "./data/annotations/panoptic_val2017"
OUTPUT_DIR = "./data/annotations/classes_val2017"

def get_categories_dict(js):
    temp_dict = {
        cate["id"]: cate["supercategory"] for cate in js["categories"]
    }
    temp_dict[0] = "unlabeled"
    return temp_dict

def process_ann(annotation, catid_to_classid):
    input_file = os.path.join(ANN_SEGMENTS_DIR, annotation["file_name"])
    output_file = os.path.join(OUTPUT_DIR, annotation["file_name"][:-4] + ".npy")

    labels = io.imread(input_file)

    labels = COCODataset.rgb2id(labels)
    class_labels = labels.copy()
    for segment in annotation["segments_info"]:
        k, v = segment["id"], segment["category_id"]
        class_labels[labels == k] = catid_to_classid[v]

    np.save(output_file, class_labels.astype(np.uint8))

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

if __name__ == "__main__":

    ann_file_path = "./data/annotations/panoptic_val2017.json"

    with open(ann_file_path, "r") as f:
        js = json.load(f)

    # cat_dict = get_categories_dict(js)
    # cat_ids = sorted(cat_dict.keys())

    # catid_to_classid = {
    #     catid: i
    #     for catid, i
    #     in zip(cat_ids, range(len(cat_ids)))}

    # with open("catid_to_classid.pkl", "wb") as f:
    #     pickle.dump(catid_to_classid, f)

    with open("catid_to_classid.pkl", "rb") as f:
        catid_to_classid = pickle.load(f)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    annotations = js["annotations"]

    with mp.Pool(10) as pool:
        args = ((ann, catid_to_classid) for ann in annotations)

        for _ in tqdm(pool.istarmap(process_ann, args), total=len(annotations)):
            pass
