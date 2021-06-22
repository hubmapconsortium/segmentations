import gc
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from deepcell.applications import MultiplexSegmentation
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import load_model
from utils import *

Image = np.ndarray
warnings.filterwarnings("ignore", category=DeprecationWarning)

default_model_path = Path("/.keras/models/MultiplexSegmentation")


class DeepcellWrapper:
    def __init__(self, model_path: Optional[Path] = default_model_path):
        self._session = None
        self._keras_model = None
        self._init_tf(model_path)
        self._model = MultiplexSegmentation(model=self._keras_model)

    def _init_tf(self, model_path: Optional[Path]):
        # setup tensorflow
        config = ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.visible_device_list = str(self.gpu_id)
        self._session = InteractiveSession(config=config)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        # load pre-downloaded keras model
        if model_path is not None:
            self._keras_model = load_model(model_path)

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
                nucleus_boundary=nuc_boundaries[i],
            )
            segmentation_output.append(img_set)
        gc.collect()
        return segmentation_output

    def close_session(self):
        self._session.close()

    def _segment_cell_and_nucleus(self, img_stack: Image) -> Image:
        return self._model.predict(img_stack, compartment="both")

    def _prepare_channels(self, img_batch: List[Dict[str, Image]]) -> Image:
        cell_channels = []
        for el in img_batch:
            cell_ch = np.stack((el["nucleus"], el["cell"]), axis=-1)
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
