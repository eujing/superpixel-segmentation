# %%
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json

# import panopticapi.utils
import os
import skimage.segmentation as segmentation
from multiprocessing import Pool

# %%
root_dir='/mnt/d/Learning/CMU/10708_Graphical_Model/Project/Segmentation/'
image_dir = "data/val2017/"
annotation_dir = "data/panoptic_annotations_trainval2017/"
superpixels_output_dir = "data/sp_segments200_compact10/val/"
annotation_type = "panoptic_val2017"

image_dir, annotation_dir, superpixels_output_dir=map( lambda x:os.path.join(root_dir, x),[image_dir, annotation_dir, superpixels_output_dir])




with open(os.path.join(annotation_dir, "annotations", f"{annotation_type}.json")) as f:
    js_val = json.load(f)

annotation_image_dir = os.path.join(
    annotation_dir, "annotations", f"{annotation_type}/"
)
# %%
# Find all neiboring superpixels
def find_sp_neighbors(sp):
    sp_center = sp[:-1, :-1, np.newaxis]
    sp_right = sp[:-1, 1:, np.newaxis]
    sp_down = sp[1:, 1:, np.newaxis]
    diff_x = np.concatenate(
        [sp_center + sp_right, np.abs(sp_center - sp_right)], axis=2
    )
    diff_y = np.concatenate([sp_center + sp_down, np.abs(sp_center - sp_down)], axis=2)
    collection_diff = np.unique(
        np.append(diff_x[diff_x[..., 1] > 0], diff_y[diff_y[..., 1] > 0], axis=0),
        axis=0,
    )
    collection_edges = np.zeros(collection_diff.shape, dtype="uint16")
    collection_edges[:, 0] = (collection_diff[:, 0] + collection_diff[:, 1]) / 2
    collection_edges[:, 1] = (collection_diff[:, 0] - collection_diff[:, 1]) / 2
    return collection_edges


# %%
def generate_sp(
    annotation,
    output_dir,
    n_segments=200,
    compactness=10.0,
    max_iter=10,
    sigma=0,
    multichannel=True,
    convert2lab=True,
    enforce_connectivity=False,
    min_size_factor=0.5,
    max_size_factor=3,
):
    try:
        file_id_str = annotation["file_name"][:-4]
        img = io.imread(os.path.join(image_dir, file_id_str + ".jpg"))
        if len(img.shape)==2:
            img=np.concatenate([img[..., np.newaxis]]*3, axis=-1)
        # img_label = io.imread(os.path.join(annotation_image_dir, annotation["file_name"]))
        superpixels_labels = segmentation.slic(
            img,
            n_segments=n_segments,
            compactness=compactness,
            max_iter=max_iter,
            sigma=sigma,
            multichannel=multichannel,
            convert2lab=convert2lab,
            enforce_connectivity=enforce_connectivity,
            min_size_factor=min_size_factor,
            max_size_factor=max_size_factor,
            start_label=0,
        )
        collection_edges = find_sp_neighbors(superpixels_labels)
        output_file = os.path.join(
            output_dir + f"{file_id_str}_sp"
        )
        np.savez(output_file, superpixels=superpixels_labels.astype('uint16'), edges=collection_edges)
    except:
        print(annotation["file_name"])
        return annotation["file_name"]


# %%
if __name__ == "__main__":
    with Pool() as p:
        temp=p.starmap(
            generate_sp,
            ((i, superpixels_output_dir) for i in js_val["annotations"]),
        )
# %%
