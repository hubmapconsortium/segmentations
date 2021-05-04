import argparse
import os
import sys
from pathlib import Path
from math import ceil
from typing import List, Dict

import numpy as np
import tifffile as tif

from utils import alpha_num_order, make_dir_if_not_exists, path_to_str
#import deepcell_wrapper
#import cellpose_wrapper
from cellpose_wrapper2 import CellposeWrapper
from deepcell_wrapper2 import DeepcellWrapper

Image = np.ndarray


def collect_img_dirs(dataset_dir: Path):
    img_dirs = [p for p in list(dataset_dir.iterdir()) if p.is_dir()]
    img_dirs = sorted(img_dirs, key=lambda path: alpha_num_order(path.name))
    img_dirs_dict = dict()
    for img_dir in img_dirs:
        img_dirs_dict[img_dir.name] = img_dir
    return img_dirs_dict


def read_img_dir(img_dir: Path) -> Dict[str, Image]:
    out_dict = dict(
        nucleus=tif.imread(path_to_str(img_dir / 'nucleus.tif')),
        cytoplasm=tif.imread(path_to_str(img_dir / 'cytoplasm.tif')),
        membrane=tif.imread(path_to_str(img_dir / 'membrane.tif'))
    )
    return out_dict


def load_img_batch(dataset_dir: Path, batch_size=10) -> Dict[str, Dict[str, Image]]:
    img_dirs = collect_img_dirs(dataset_dir)
    dir_names = list(img_dirs.keys())

    n_batches = ceil(len(img_dirs) / batch_size)
    for b in range(0, n_batches):
        f = b * batch_size
        t = f + batch_size
        if t > len(img_dirs):
            t = len(img_dirs)

        batch = dict()
        for i in range(f, t):
            dir_path = img_dirs[dir_names[i]]
            batch[dir_names[i]] = read_img_dir(dir_path)
        yield batch


def main(modality: str, dataset_dir: Path):
    out_dir = Path("/output/")
    out_dir_deepcell = out_dir / "deepcell"
    out_dir_cellpose = out_dir / "cellpose"
    make_dir_if_not_exists(out_dir_deepcell)
    make_dir_if_not_exists(out_dir_cellpose)

    img_batch_gen = load_img_batch(dataset_dir, 10)

    deepcell_segm = DeepcellWrapper()
    cellpose_segm = CellposeWrapper()

    img_batch = next(img_batch_gen)
    deepcell_segm.segment(img_batch, out_dir_deepcell)
    cellpose_segm.segment(img_batch, out_dir_cellpose)


#
# def check_if_segm_imgs_in_dir(dir_path: Path):
#     names = ('nucleus', 'membrane', 'cytoplasm')
#     listing = list(dir_path.iterdir())
#     if listing == []:
#         return listing
#     else:
#         found_channels = dict()
#         for path in listing:
#             path_str = path_to_str(path)
#             found = {name: path_str for name in names if name in path_str}
#             if found != {}:
#                 found_channels.update(found)
#         if len(found_channels.keys()) == len(names):
#             return True
#         else:
#             print('Not all channels found in', dir_path,
#                   'Found channels', found_channels)
#             return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, help="mibi or celldive")
    parser.add_argument("--dataset_dir", type=Path, help="path to directory with images")
    args = parser.parse_args()

    main(args.modality, args.dataset_dir)
