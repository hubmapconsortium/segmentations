from typing import List, Dict, Tuple
import gc

import numpy as np
import tensorflow as tf
from deepcell.applications import MultiplexSegmentation
from skimage.io import imread
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from utils import *

Image = np.ndarray


class DeepcellWrapper:
    def __init__(self):
        self._init_tf()
        self._model = MultiplexSegmentation()

    def _init_tf(self):
        # setup tensorflow
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    def segment(self, img_batch: List[Dict[str, Image]]) -> List[Dict[str, Image]]:
        cell_channels = self._prepare_channels(img_batch)
        gc.collect()
        masks = self._segment_cell_and_nucleus(cell_channels)
        cell_masks, nuc_masks = self._separate_batch(masks)
        cell_boundaries = get_boundary(cell_masks)
        nuc_boundaries = get_boundary(nuc_masks)
        gc.collect()
        batch_size = len(img_batch)
        segmentation_output = []
        for i in range(0, batch_size):
            img_set = dict(
                cell=cell_masks[i],
                nucleus=nuc_masks[i],
                cell_boundary=cell_boundaries[i],
                nucleus_boundary=nuc_boundaries[i]
            )
            segmentation_output.append(img_set)
        gc.collect()
        return segmentation_output

    def reset_cell_and_nuc_size(self):
        pass

    def segment_tiled(self, tile_batch: List[Dict[str, List[Image]]]) -> List[Dict[str, List[Image]]]:
        tiled_imgs = self._prepare_channels_tiled(tile_batch)
        gc.collect()
        tiled_masks: List[Image] = []
        for img in tiled_imgs:
            mask = self._segment_cell_and_nucleus(img)
            tiled_masks.append(mask)
        gc.collect()
        cell_masks_tiled = []
        nuc_masks_tiled = []
        cell_b_tiled = []
        nuc_b_tiled = []
        for mask in tiled_masks:
            cell_masks, nuc_masks = self._separate_batch(mask)
            cell_boundaries = get_boundary(cell_masks)
            nuc_boundaries = get_boundary(nuc_masks)
            cell_masks_tiled.append(cell_masks)
            nuc_masks_tiled.append(nuc_masks)
            cell_b_tiled.append(cell_boundaries)
            nuc_b_tiled.append(nuc_boundaries)
            gc.collect()
        batch_size = len(tile_batch)
        segmentation_output = []
        for i in range(0, batch_size):
            img_set = dict(
                cell=cell_masks_tiled[i],
                nucleus=nuc_masks_tiled[i],
                cell_boundary=cell_b_tiled[i],
                nucleus_boundary=nuc_b_tiled[i]
            )
            segmentation_output.append(img_set)
        gc.collect()
        return segmentation_output

    def _segment_cell_and_nucleus(self, img_stack: Image) -> Image:
        return self._model.predict(img_stack, compartment="both")

    def _prepare_channels(self, img_batch: List[Dict[str, Image]]) -> Image:
        cell_channels = []
        for el in img_batch:
            cell_ch = np.stack((el['nucleus'], el['cell']), axis=-1)
            cell_ch = np.expand_dims(cell_ch, 0)
            cell_channels.append(cell_ch)
        return np.concatenate(cell_channels, axis=0)

    def _stack_tiles_from_different_channels(self, tiles: List[Image]):
        return np.expand_dims(np.stack(tiles, axis=-1), 0)

    def _prepare_channels_tiled(self, img_batch: List[Dict[str, List[Image]]]) -> List[Image]:
        tile_batches = []
        for el in img_batch:
            img_tiles = []
            num_tiles = len(el['nucleus'])
            for i in range(0, num_tiles):
                tile_pair = [el['nucleus'][i], el['cell'][i]]
                img_tiles = self._stack_tiles_from_different_channels(tile_pair)
            tile_batch = np.concatenate(img_tiles, axis=0)
            tile_batches.append(tile_batch)
        return tile_batches

    def _separate_batch(self, img_stack: Image) -> Tuple[List[Image], List[Image]]:
        n_imgs = img_stack.shape[0]
        cell_masks = []
        nuc_masks = []
        for i in range(0, n_imgs):
            cell_mask = img_stack[i, :, :, 0]
            nuc_mask = img_stack[i, :, :, 1]
            cell_masks.append(cell_mask)
            nuc_masks.append(nuc_mask)
        return cell_masks, nuc_masks
