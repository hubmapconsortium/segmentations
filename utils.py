# utils.py>
import os
from os.path import join
from pathlib import Path

import numpy as np
import tifffile as tif
from skimage.segmentation import find_boundaries


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def get_boundary(mask):
    mask_boundary = find_boundaries(mask)
    mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
    return mask_boundary_indexed


def save_segmentation_masks(out_dir, cell, nuc, cell_b, nuc_b):
    tif.imwrite(path_to_str(out_dir / "cell.tif"), cell.astype(np.uint32))
    tif.imwrite(path_to_str(out_dir / "nucleus.tif"), nuc.astype(np.uint32))
    tif.imwrite(path_to_str(out_dir / "cell_boundaries.tif"), cell_b.astype(np.uint32))
    tif.imwrite(path_to_str(out_dir / "nucleus_boundaries.tif"), nuc_b.astype(np.uint32))
