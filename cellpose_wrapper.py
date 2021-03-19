import os
import sys
import time
from os.path import join

import numpy as np
from cellpose import models
from skimage.io import imread, imsave

sys.path.append(os.getcwd())
from cellpose import utils

from utils import *

if __name__ == "__main__":
    use_GPU = utils.use_gpu()
    print("GPU activated? %d" % use_GPU)

    file_dir = sys.argv[1]
    im1 = imread(join(file_dir, "cytoplasm.tif"))
    im2 = imread(join(file_dir, "nucleus.tif"))
    im = np.stack((im1, im2))

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
    cell_boundary_mask = get_boundary(cell_mask)

    # get nucleus masks
    model_nuc = models.Cellpose(gpu=use_GPU, model_type="nuclei")
    channels = [[0, 0]]
    nuc_mask, flows, styles, diams = model_nuc.eval(
        im2, diameter=None, flow_threshold=None, channels=channels
    )
    nuc_boundary_mask = get_boundary(nuc_mask)
    save_dir = join(file_dir, "cellpose")
    os.makedirs(save_dir)
    save_ome_tiff(save_dir, cell_mask, nuc_mask, cell_boundary_mask, nuc_boundary_mask)
