import argparse
import copy
import os
import os.path as osp
from typing import List

import dask
import numpy as np
import tifffile as tif

Image = np.ndarray



def generate_slicer_info(original_img_shape: List[int], tile_shape_no_overlap: List[int], overlap: int):
    img_height, img_width = original_img_shape
    tile_h, tile_w = tile_shape_no_overlap

    padding = dict(left=0, right=0, top=0, bottom=0)
    if img_width % tile_w == 0:
        padding["right"] = 0
    else:
        padding["right"] = tile_w - (img_width % tile_w)
    if img_height % tile_h == 0:
        padding["bottom"] = 0
    else:
        padding["bottom"] = tile_h - (img_height % tile_h)

    if img_width <= tile_w:
        x_ntiles = 1
    else:
        x_ntiles = img_width // tile_w if img_width % tile_w == 0 else (img_width // tile_w) + 1
    if img_height <= tile_h:
        y_ntiles = 1
    else:
        y_ntiles = img_height // tile_h if img_height % tile_h == 0 else (img_height // tile_h) + 1

    slicer_info = dict()
    slicer_info["padding"] = padding
    slicer_info["overlap"] = overlap
    slicer_info["num_tiles"] = {"x": x_ntiles, "y": y_ntiles}
    slicer_info["tile_shape_no_overlap"] = {"x": tile_w, "y": tile_h}
    slicer_info["tile_shape_with_overlap"] = {
        "x": tile_w + overlap * 2,
        "y": tile_h + overlap * 2,
    }
    slicer_info["dtype"] = np.uint32
    return slicer_info


def get_tile(arr, hor_f: int, hor_t: int, ver_f: int, ver_t: int, overlap=0):
    hor_f -= overlap
    hor_t += overlap
    ver_f -= overlap
    ver_t += overlap

    left_check = hor_f
    top_check = ver_f
    right_check = hor_t - arr.shape[1]
    bot_check = ver_t - arr.shape[0]

    left_pad_size = 0
    top_pad_size = 0
    right_pad_size = 0
    bot_pad_size = 0

    if left_check < 0:
        left_pad_size = abs(left_check)
        hor_f = 0
    if top_check < 0:
        top_pad_size = abs(top_check)
        ver_f = 0
    if right_check > 0:
        right_pad_size = right_check
        hor_t = arr.shape[1]
    if bot_check > 0:
        bot_pad_size = bot_check
        ver_t = arr.shape[0]

    tile_slice = (slice(ver_f, ver_t), slice(hor_f, hor_t))
    tile = arr[tile_slice]
    padding = ((top_pad_size, bot_pad_size), (left_pad_size, right_pad_size))
    if max(padding) > (0, 0):
        tile = np.pad(tile, padding, mode="constant")
    return tile


def split_img(
    arr: Image,
    tile_h: int,
    tile_w: int,
    overlap: int
) -> List[Image]:
    """Splits image into tiles by size of tile.
    tile_w - tile width
    tile_h - tile height
    """
    x_axis = -1
    y_axis = -2
    arr_width, arr_height = arr.shape[x_axis], arr.shape[y_axis]

    if arr_width <= tile_w:
        x_ntiles = 1
    else:
        x_ntiles = arr_width // tile_w if arr_width % tile_w == 0 else (arr_width // tile_w) + 1
    if arr_height <= tile_h:
        y_ntiles = 1
    else:
        y_ntiles = arr_height // tile_h if arr_height % tile_h == 0 else (arr_height // tile_h) + 1

    tiles = []
    # row
    for i in range(0, y_ntiles):
        # height of this tile
        ver_f = tile_h * i
        ver_t = ver_f + tile_h

        # col
        for j in range(0, x_ntiles):
            # width of this tile
            hor_f = tile_w * j
            hor_t = hor_f + tile_w

            tile = get_tile(arr, hor_f, hor_t, ver_f, ver_t, overlap)
            tiles.append(tile)

    slicer_info = generate_slicer_info(arr.shape, [tile_h, tile_w], overlap)
    return tiles, slicer_info
