from pathlib import Path
from typing import List, Dict
import gc

import numpy as np
from cellpose import models

from utils import *

Image = np.ndarray


class CellposeWrapper:
    def __init__(self):
        self._gpu = models.use_gpu()
        self._model_cyto = models.Cellpose(gpu=self._gpu, model_type="cyto")
        self._model_nuc = models.Cellpose(gpu=self._gpu, model_type="nuclei")

    def segment(self, batch_img_dict: Dict[str, Dict[str, Image]], out_dir: Path):
        cell_channels, nuc_channels = self._prepare_channels(batch_img_dict)
        gc.collect()
        cell_masks = self._segment_cell(cell_channels)
        nuc_masks = self._segment_nucleus(nuc_channels)
        cell_boundaries = get_boundary(cell_masks)
        nuc_boundaries = get_boundary(nuc_masks)
        gc.collect()
        img_dirs = list(batch_img_dict.keys())
        out_dirs = [out_dir / img_dir for img_dir in img_dirs]
        self._save_masks(out_dirs, cell_masks, nuc_masks, cell_boundaries, nuc_boundaries)
        gc.collect()

    def _segment_cell(self, img_stack_list: List[Image]) -> List[Image]:
        cell_channels = [[0, 1]] * len(img_stack_list)
        cell_mask, flows, styles, diams = self._model_cyto.eval(
            img_stack_list,
            diameter=None,
            flow_threshold=None,
            channels=cell_channels
        )
        if not isinstance(cell_mask, list):
            return [cell_mask]
        else:
            return cell_mask

    def _segment_nucleus(self, img_list: List[Image]) -> List[Image]:
        nuc_channels = [[0, 0]] * len(img_list)
        nuc_mask, flows, styles, diams = self._model_nuc.eval(
            img_list,
            diameter=None,
            flow_threshold=None,
            channels=nuc_channels
        )
        if not isinstance(nuc_mask, list):
            return [nuc_mask]
        else:
            return nuc_mask

    def _save_masks(self, out_dirs, cell_masks, nuc_masks, cell_boundaries, nuc_boundaries):
        for i in range(0, len(out_dirs)):
            save_segmentation_masks(out_dirs[i],
                                    cell_masks[i],
                                    nuc_masks[i],
                                    cell_boundaries[i],
                                    nuc_boundaries[i]
                                    )

    def _prepare_channels(self, batch_img_dict: Dict[str, Dict[str, Image]]):
        cell_channels = []
        nuc_channels = []
        for batch, imgs in batch_img_dict.items():
            cell_ch = np.stack((imgs['cytoplasm'], imgs['nucleus']), axis=0)
            cell_channels.append(cell_ch)
            nuc_channels.append(imgs['nucleus'])
        return cell_channels, nuc_channels
