from typing import List, Dict, Tuple, Union, Any
import gc
from collections import Counter
import numpy as np
from cellpose import models

from utils import *

Image = np.ndarray
Ch_name = str
Tiles = List[Image]


class CellposeWrapper:
    def __init__(self):
        self._gpu = models.use_gpu()
        self._model_cyto = models.Cellpose(gpu=self._gpu, model_type="cyto")
        self._model_nuc = models.Cellpose(gpu=self._gpu, model_type="nuclei")
        self._cell_size = None
        self._nuc_size = None

    def segment(self, img_batch: List[Dict[str, Image]]) -> List[Dict[str, Image]]:
        cell_channels, nuc_channels = self._prepare_channels(img_batch)
        gc.collect()
        cell_masks = self._segment_cell(cell_channels)
        nuc_masks = self._segment_nucleus(nuc_channels)
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
        self._cell_size = None
        self._nuc_size = None

    def process_diams(self, diams: List[float]) -> Union[float, Any]:
        """ If there is more than 1 cell diameter produced by cellpose will
            take only those that constitute at least 10% of number of diameters
            and take average of them
        """
        if isinstance(diams, list):
            num_diams = len(diams)
            if num_diams == 1:
                return diams
            else:
                rounded_diams = np.round(diams)
                if num_diams > 10:
                    frac10 = num_diams // 10
                else:
                    frac10 = 1
                counts = Counter(rounded_diams)
                filtered_diams = []
                for val, num in counts.items():
                    if num >= frac10:
                        filtered_diams.extend([val] * num)
                mean_diam = np.mean(filtered_diams)
                return mean_diam
        else:
            return diams

    def _segment_cell(self, img_stack_list: List[Image]) -> List[Image]:
        cell_channels = [[0, 1]] * len(img_stack_list)
        cell_mask, flows, styles, diams = self._model_cyto.eval(
            img_stack_list,
            diameter=self._cell_size,
            flow_threshold=None,
            channels=cell_channels,
            tile=True,
            tile_overlap=0.1
        )
        if self._cell_size is None:
            self._cell_size = self.process_diams(diams)
        if not isinstance(cell_mask, list):
            return [cell_mask]
        else:
            return cell_mask

    def _segment_nucleus(self, img_list: List[Image]) -> List[Image]:
        nuc_channels = [[0, 0]] * len(img_list)
        nuc_mask, flows, styles, diams = self._model_nuc.eval(
            img_list,
            diameter=self._nuc_size,
            flow_threshold=None,
            channels=nuc_channels,
            tile=True,
            tile_overlap=0.1
        )
        if self._nuc_size is None:
            self._nuc_size = self.process_diams(diams)
        if not isinstance(nuc_mask, list):
            return [nuc_mask]
        else:
            return nuc_mask

    def segment_tiled(self, tile_batch: List[Dict[Ch_name, Tiles]]) -> List[Dict[Ch_name, Tiles]]:
        cell_channels_tiled, nuc_channels_tiled = self._prepare_channels_tiled(tile_batch)
        gc.collect()
        cell_masks_tiled = []
        nuc_masks_tiled = []
        for i in range(0, len(cell_channels_tiled)):
            cell_mask = self._segment_cell(cell_channels_tiled[i])
            nuc_mask = self._segment_nucleus(nuc_channels_tiled[i])
            cell_masks_tiled.append(cell_mask)
            nuc_masks_tiled.append(nuc_mask)
        gc.collect()
        cell_b_tiled = []
        nuc_b_tiled = []
        for i in range(0, len(cell_masks_tiled)):
            cell_boundaries = get_boundary(cell_masks_tiled[i])
            nuc_boundaries = get_boundary(nuc_channels_tiled[i])
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

    def _prepare_channels(self, img_batch: List[Dict[str, Image]]) -> Tuple[List[Image], List[Image]]:
        cell_channels = []
        nuc_channels = []
        for el in img_batch:
            cell_ch = np.stack((el['cell'], el['nucleus']), axis=0)
            cell_channels.append(cell_ch)
            nuc_channels.append(el['nucleus'])
        return cell_channels, nuc_channels

    def _prepare_channels_tiled(self, img_batch: List[Dict[Ch_name, Tiles]]) -> Tuple[List[Tiles], List[Tiles]]:
        cell_channels_tiled = []
        nuc_channels_tiled = []
        for el in img_batch:
            cell_tiles = []
            nuc_tiles = []
            num_tiles = len(el['nucleus'])
            for i in range(0, num_tiles):
                cell_tiles.append(np.stack((el['cell'][i], el['nucleus'][i]), axis=0))
                nuc_tiles.append(el['nucleus'][i])
            cell_channels_tiled.append(cell_tiles)
            nuc_channels_tiled.append(nuc_tiles)
        return cell_channels_tiled, nuc_channels_tiled
