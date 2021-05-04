import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from deepcell.applications import MultiplexSegmentation
from skimage.io import imread
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from utils import *


def main(img_dir: Path, out_dir: Path):
    # setup tensorflow
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # setup deepcell
    app = MultiplexSegmentation()

    # read nucleus and membrane channels
    im1 = imread(path_to_str(img_dir / "nucleus.tif"))
    im2 = imread(path_to_str(img_dir / "membrane.tif"))
    im = np.stack((im1, im2), axis=-1)
    im = np.expand_dims(im, 0)

    # predict the masks of nucleus and the whole cell
    labeled_image = app.predict(im, compartment="both")
    labeled_image = np.squeeze(labeled_image, axis=0)
    cell_mask = labeled_image[:, :, 0]
    nuc_mask = labeled_image[:, :, 1]

    # get the mask boundaries
    cell_boundary_mask = get_boundary(cell_mask)
    nuc_boundary_mask = get_boundary(nuc_mask)
    session.close()

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
