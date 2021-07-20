import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from utils import *

Image = np.ndarray
Ch_name = str
Tiles = List[Image]

default_model_path = Path("/opt")


class CellposeWrapper:
    def __init__(self, model_path: Optional[Path] = default_model_path):
        with home_dir_env_override(model_path):
            from cellpose import models

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
                nucleus_boundary=nuc_boundaries[i],
            )
            segmentation_output.append(img_set)
        gc.collect()
        return segmentation_output

    def close_session(self):
        pass

    def _segment_cell(self, img_stack_list: List[Image]) -> List[Image]:
        cell_channels = [[0, 1]] * len(img_stack_list)
        cell_mask, flows, styles, diams = self._model_cyto.eval(
            img_stack_list,
            diameter=self._cell_size,
            flow_threshold=None,
            channels=cell_channels,
            tile=True,
            tile_overlap=0.1,
            batch_size=100,
        )
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
            tile_overlap=0.1,
            batch_size=100,
        )
        if not isinstance(nuc_mask, list):
            return [nuc_mask]
        else:
            return nuc_mask

    def _prepare_channels(
        self, img_batch: List[Dict[str, Image]]
    ) -> Tuple[List[Image], List[Image]]:
        cell_channels = []
        nuc_channels = []
        for el in img_batch:
            cell_ch = np.stack((el["cell"], el["nucleus"]), axis=0)
            cell_channels.append(cell_ch)
            nuc_channels.append(el["nucleus"])
        return cell_channels, nuc_channels
