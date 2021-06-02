from math import ceil
from pathlib import Path
from typing import Tuple, Dict, List, Union, Iterator
import re
from itertools import chain

import numpy as np
import tifffile as tif

from utils import alpha_num_order, get_img_listing, path_to_str

from img_proc.slicer import split_img
Image = np.ndarray


class BatchLoader:
    def __init__(self):
        self.batch_size = 10
        self.dataset_dir = Path('.')
        self.segmentation_channel_names = ("nucleus", "cell")
        self.split_img_into_tiles = False
        self._slicer_info = dict()
        self._img_batch_gen = None

    def init_img_batch_generator_cross_dir(self):
        """ Will produce batches across img directories """
        img_dirs = self.collect_img_dirs(self.dataset_dir)
        img_info, img_sets = self.get_img_sets(img_dirs, self.segmentation_channel_names)
        self._img_batch_gen = self.create_img_batch_gen(img_info, img_sets, self.batch_size)

    def init_img_batch_generator_per_dir(self):
        """ Will produce batches only inside one img directory at a time """
        img_dirs = self.collect_img_dirs(self.dataset_dir)
        generators_per_dir = []
        for img_dir, img_path in img_dirs.items():
            img_info, img_sets = self.get_img_sets({img_dir: img_path}, self.segmentation_channel_names)
            this_dir_gen = self.create_img_batch_gen(img_info, img_sets, self.batch_size)
            generators_per_dir.append(this_dir_gen)
        self._img_batch_gen = chain(*generators_per_dir)  # chain generators into one

    def get_img_batch(self):
        try:
            info_batch, img_batch = next(self._img_batch_gen)
            return info_batch, img_batch
        except StopIteration:
            return None, None

    def get_img_batch_tiled(self) -> Tuple[List[Dict[str, str]], List[dict], List[Dict[str, List[Image]]]]:
        try:
            info_batch, img_batch = next(self._img_batch_gen)
            tile_info, tile_batch = self.split_batch_into_tiles(info_batch, img_batch)
            return info_batch, tile_info, tile_batch
        except StopIteration:
            return None, None, None

    def split_batch_into_tiles(self, info_batch, img_batch: List[Dict[str, Image]]) -> Tuple[List[dict], List[Dict[str, List[Image]]]]:
        tile_height, tile_width = (1000, 1000)
        overlap = 100
        tile_batch = []
        tile_info = []
        for img_set in img_batch:
            tile_set = dict()
            for channel, img in img_set.items():
                tiles, slicer_info = split_img(img, tile_height, tile_width, overlap)
                tile_set[channel] = tiles
            tile_batch.append(tile_set)
            tile_info.append(slicer_info)
        return tile_info, tile_batch

    def get_img_name_parts(self, img_name: str, segm_channel_names: Tuple[str]) -> Tuple[Union[str, None], Union[str, None]]:
        img_prefix, identified_channel = None, None
        for segm_ch_name in segm_channel_names:
            channel_name_in_file = re.search(segm_ch_name, img_name, flags=re.IGNORECASE)
            if channel_name_in_file is not None and channel_name_in_file != '':
                identified_channel = segm_ch_name
                channel_pattern = '_?' + segm_ch_name + r'\.tif'
                img_prefix = re.sub(channel_pattern, '', img_name, flags=re.IGNORECASE)
        return img_prefix, identified_channel


    def check_all_channels_present(self,
                                   img_dir: Path,
                                   img_set: Dict[str, Path],
                                   segm_channel_names: Tuple[str]
                                   ) -> bool:
        img_set_ch_names = list(img_set.keys())
        found_channels = []
        missing_channels = []
        for ch_name in segm_channel_names:
            if ch_name in img_set_ch_names:
                found_channels.append(ch_name)
            else:
                missing_channels.append(ch_name)
        if missing_channels != []:
            msg = 'Missing channels: ' + str(missing_channels) + ' from ' + str(img_dir)
            raise ValueError(msg)
        else:
            return True


    def get_dir_listing(self, img_dir: Path, segm_channel_names: Tuple[str]) -> Dict[str, Dict[str, Path]]:
        """ output {img_set: {channel_name: Image, ...}}
            img_set is defined by img_prefix,
            e.g. dataset_name_nucleus.tif
            img_set = img_prefix = dataset_name
            channel_name = nucleus
        """
        listing = get_img_listing(img_dir)
        out_dict = dict()

        for img_path in listing:
            img_prefix, channel_name = self.get_img_name_parts(img_path.name, segm_channel_names)
            if channel_name is not None:
                if img_prefix in out_dict:
                    out_dict[img_prefix][channel_name] = img_path
                else:
                    out_dict[img_prefix] = {channel_name: img_path}
        for img_set in out_dict:
            self.check_all_channels_present(img_dir, out_dict[img_set], segm_channel_names)
        return out_dict


    def get_img_sets(self, img_dirs: Dict[str, Path],
                     segm_channel_names: Tuple[str]
                     ) -> Tuple[List[Dict[str, str]], List[Dict[str, Path]]]:
        all_img_sets = []
        all_img_info = []
        for dir_name, dir_path in img_dirs.items():
            img_dir_listing = self.get_dir_listing(dir_path, segm_channel_names)
            this_dir_sets = list(img_dir_listing.keys())
            for img_set in this_dir_sets:
                all_img_sets.append(img_dir_listing[img_set])
                all_img_info.append({dir_name: img_set})

        return all_img_info, all_img_sets


    def collect_img_dirs(self, dataset_dir: Path) -> Dict[str, Path]:
        img_dirs = [p for p in list(dataset_dir.iterdir()) if p.is_dir()]
        img_dirs = sorted(img_dirs, key=lambda path: alpha_num_order(path.name))
        img_dirs_dict = dict()
        for img_dir in img_dirs:
            img_dirs_dict[img_dir.name] = img_dir
        return img_dirs_dict

    def create_img_batch_gen(self,
                             img_info: List[Dict[str, str]],
                             img_sets: List[Dict[str, Path]],
                             batch_size=10
                             ) -> Iterator[Tuple[List[Dict[str, str]], List[Dict[str, Image]]]]:
        """ output
           [ {dir_name: img_set_name} ],
           [ {channel : Image} ]
        """
        num_sets = len(img_sets)
        num_batches = ceil(num_sets / batch_size)
        print('Batch size is:', batch_size)
        for b in range(0, num_batches):
            print('Loading image batch:', b + 1, '/', num_batches)
            f = b * batch_size
            t = f + batch_size
            if t > num_sets:
                t = num_sets
            path_batch = img_sets[f:t]
            info_batch = img_info[f:t]

            img_batch = []
            for el in path_batch:
                img_el = dict()
                for ch_name, ch_path in el.items():
                    img_el[ch_name] = tif.imread(path_to_str(ch_path))
                img_batch.append(img_el)
            yield info_batch, img_batch

