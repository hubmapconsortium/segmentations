import argparse
from pathlib import Path
from math import ceil
from typing import List, Dict, Tuple
import re
from datetime import datetime

import numpy as np

from utils import make_dir_if_not_exists, write_stack_to_file
from img_proc.mask_stitcher import stitch_mask_tiles
from batch import BatchLoader

Image = np.ndarray


def save_masks(base_out_dir: Path, base_img_name: str, info: List[Dict[str, str]], imgs: List[Dict[str, Image]]):
    for i, el in enumerate(info):
        dir_name, img_set = list(el.items())[0]
        out_dir = base_out_dir / dir_name
        make_dir_if_not_exists(out_dir)
        img_name = img_set + '_' + base_img_name
        channels = imgs[i]
        mask_stack = np.stack([channels["cell"],
                               channels["nucleus"],
                               channels["cell_boundary"],
                               channels["nucleus_boundary"]
                               ],
                              axis=0)
        write_stack_to_file(out_dir, img_name, mask_stack)


def get_segmentation_method(method: str):
    if method == 'cellpose':
        from cellpose_wrapper2 import CellposeWrapper
        segmenter = CellposeWrapper()
    elif method == 'deepcell':
        from deepcell_wrapper2 import DeepcellWrapper
        segmenter = DeepcellWrapper()
    else:
        msg = 'Incorrect segmentation method ' + method
        raise ValueError(msg)
    print('Using segmentation method ' + method)
    return segmenter


def main(method: str, dataset_dir: Path, batch_size: int, use_tiles: False):
    segm_channel_names = ("nucleus", "cell")
    out_base_img_name = "mask.ome.tiff"
    out_base_dir = Path("/output/")

    start = datetime.now()
    print('Started ' + str(start))

    batcher = BatchLoader()
    batcher.batch_size = batch_size
    batcher.dataset_dir = dataset_dir
    batcher.segmentation_channel_names = segm_channel_names
    batcher.init_img_batch_generator_per_dir()

    segmenter = get_segmentation_method(method)

    if use_tiles:
        while True:
            info_batch, tile_info, tile_batch = batcher.get_img_batch_tiled()
            if info_batch is None and tile_batch is None:
                break
            segmented_tiles = segmenter.segment_tiled(tile_batch)
            for img_set_id, tiles in enumerate(segmented_tiles):
                stitched_mask = stitch_mask_tiles(tiles, tile_info[img_set_id])
                save_masks(out_base_dir, out_base_img_name, info_batch, [stitched_mask])
    else:
        while True:
            info_batch, img_batch = batcher.get_img_batch()
            if info_batch is None and img_batch is None:
                break
            segmented_batch = segmenter.segment(img_batch)
            save_masks(out_base_dir, out_base_img_name, info_batch, segmented_batch)

    fin = datetime.now()
    print('Finished ' + str(fin))
    print('Time elapsed ' + str(fin - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str,
                        help="segmentation method cellpose or deepcell")
    parser.add_argument("--dataset_dir", type=Path,
                        help="path to directory with images")
    parser.add_argument("--batch_size", default=10, type=int,
                        help="number of images to process simultaneously")
    # can't work yet because of mismatch in number of labels for cell and nuclei
    # parser.add_argument("--use_tiles", action='store_true',
    #                     help="split images into 1000x1000px tiles with 100px overlap and run segmentation")
    args = parser.parse_args()

    main(args.method, args.dataset_dir, args.batch_size, args.use_tiles)
