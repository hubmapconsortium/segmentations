import argparse
import os
import sys
import time

import numpy as np
from cellpose import models
from skimage.io import imread, imsave

from utils import *


def main(img_dir: Path, out_dir: Path):
    use_GPU = models.use_gpu()
    print("GPU activated? %d" % use_GPU)

    im1 = imread(path_to_str(img_dir / "cytoplasm.tif"))
    im2 = imread(path_to_str(img_dir / "nucleus.tif"))
    im = np.stack((im1, im2), axis=0)

    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model_cyto = models.Cellpose(gpu=use_GPU, model_type="cyto")

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # or if you have different types of channels in each image
    channels = [[0, 1]]

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended)
    # diameter can be a list or a single number for all images

    cell_mask, flows, styles, diams = model_cyto.eval(
        im, diameter=None, flow_threshold=None, channels=channels
    )

    # get nucleus masks
    model_nuc = models.Cellpose(gpu=use_GPU, model_type="nuclei")
    channels = [[0, 0]]
    nuc_mask, flows, styles, diams = model_nuc.eval(
        im2, diameter=None, flow_threshold=None, channels=channels
    )

    cell_boundary_mask = get_boundary(cell_mask)
    nuc_boundary_mask = get_boundary(nuc_mask)

    save_segmentation_masks(out_dir, cell_mask, nuc_mask, cell_boundary_mask, nuc_boundary_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=Path,
        help="path to directory with images nucleus.tif, cytoplasm.tif, membrane.tif",
    )
    parser.add_argument(
        "--out_dir", type=Path, help="path to directory to output segmentation masks"
    )
    args = parser.parse_args()

    main(args.img_dir, args.out_dir)
