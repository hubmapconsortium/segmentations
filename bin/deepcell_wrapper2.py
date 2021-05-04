from pathlib import Path
import numpy as np
import tensorflow as tf
from deepcell.applications import MultiplexSegmentation
from skimage.io import imread
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from utils import *

from typing import List, Dict, Tuple
import gc

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

    def segment(self, batch_img_dict: Dict[str, Dict[str, Image]], out_dir: Path):
        cell_channels = self._prepare_channels(batch_img_dict)
        gc.collect()
        masks = self._segment_cell_and_nucleus(cell_channels)
        cell_masks, nuc_masks = self._separate_batch(masks)
        cell_boundaries = get_boundary(cell_masks)
        nuc_boundaries = get_boundary(nuc_masks)
        gc.collect()
        img_dirs = list(batch_img_dict.keys())
        out_dirs = [out_dir / img_dir for img_dir in img_dirs]
        self._save_masks(out_dirs, cell_masks, nuc_masks, cell_boundaries, nuc_boundaries)
        gc.collect()

    def _segment_cell_and_nucleus(self, img_stack: Image) -> Image:
        return self._model.predict(img_stack, compartment="both")

    def _save_masks(self, out_dirs, cell_masks, nuc_masks, cell_boundaries, nuc_boundaries):
        for i in range(0, len(out_dirs)):
            save_segmentation_masks(out_dirs[i],
                                    cell_masks[i],
                                    nuc_masks[i],
                                    cell_boundaries[i],
                                    nuc_boundaries[i]
                                    )

    def _prepare_channels(self, batch_img_dict: Dict[str, Dict[str, Image]]) -> Image:
        cell_channels = []
        for batch, imgs in batch_img_dict.items():
            cell_ch = np.stack((imgs['nucleus'], imgs['membrane']), axis=-1)
            cell_ch = np.expand_dims(cell_ch, 0)
            cell_channels.append(cell_ch)
        return np.concatenate(cell_channels, axis=0)

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
