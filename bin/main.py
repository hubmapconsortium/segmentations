import argparse
from pathlib import Path
from math import ceil
from typing import List, Dict, Tuple
import re
from datetime import datetime

import numpy as np
import tifffile as tif

from utils import alpha_num_order, make_dir_if_not_exists, path_to_str, get_img_listing, write_stack_to_file

Image = np.ndarray


def get_img_name_parts(img_name: str, segm_channel_names: Tuple[str]) -> Tuple[str, str]:
    identified_channel = ''
    img_prefix = ''
    for ch_name in segm_channel_names:
        if re.search(ch_name, img_name, flags=re.IGNORECASE) is not None:
            identified_channel = ch_name
            channel_pattern = '_?' + ch_name + r'\.tif'
            img_prefix = re.sub(channel_pattern, '', img_name, flags=re.IGNORECASE)
    return img_prefix, identified_channel


def check_all_channels_present(img_dir: Path, img_set: Dict[str, Path], segm_channel_names: Tuple[str]) -> bool:
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


def get_dir_listing(img_dir: Path, segm_channel_names: Tuple[str]) -> Dict[str, Dict[str, Path]]:
    """ output {img_set: {channel_name: Image, ...}}
        img_set is defined by img_prefix,
        e.g. dataset_name_nucleus.tif
        img_prefix = img_set = dataset_name
        channel_name = nucleus
    """
    listing = get_img_listing(img_dir)
    out_dict = dict()

    for img_path in listing:
        img_prefix, channel_name = get_img_name_parts(img_path.name, segm_channel_names)
        if img_prefix in out_dict:
            out_dict[img_prefix][channel_name] = img_path
        else:
            out_dict[img_prefix] = {channel_name: img_path}
    for img_set in out_dict:
        check_all_channels_present(img_dir, out_dict[img_set], segm_channel_names)
    return out_dict


def collect_img_dirs(dataset_dir: Path) -> Dict[str, Path]:
    img_dirs = [p for p in list(dataset_dir.iterdir()) if p.is_dir()]
    img_dirs = sorted(img_dirs, key=lambda path: alpha_num_order(path.name))
    img_dirs_dict = dict()
    for img_dir in img_dirs:
        img_dirs_dict[img_dir.name] = img_dir
    return img_dirs_dict


def get_img_sets(img_dirs: Dict[str, Path],
                 segm_channel_names: Tuple[str]
                 ) -> Tuple[List[Dict[str, str]], List[Dict[str, Path]]]:
    all_img_sets = []
    all_img_info = []
    for dir_name, dir_path in img_dirs.items():
        img_dir_listing = get_dir_listing(dir_path, segm_channel_names)
        this_dir_sets = list(img_dir_listing.keys())
        for img_set in this_dir_sets:
            all_img_sets.append(img_dir_listing[img_set])
            all_img_info.append({dir_name: img_set})

    return all_img_info, all_img_sets


def load_img_batch(dataset_dir: Path,
                   segm_channel_names: Tuple[str],
                   batch_size=10
                   ) -> Tuple[List[Dict[str, str]], List[Dict[str, Path]]]:
    """ output dict {img_dir : {img_set : {channel : Image}}}"""
    img_dirs = collect_img_dirs(dataset_dir)
    all_img_info, all_img_sets = get_img_sets(img_dirs, segm_channel_names)
    num_sets = len(all_img_sets)
    n_batches = ceil(num_sets / batch_size)
    print('Batch size is:', batch_size)
    for b in range(0, n_batches):
        print('Loading image batch:', b + 1, '/', n_batches)
        f = b * batch_size
        t = f + batch_size
        if t > num_sets:
            t = num_sets
        path_batch = all_img_sets[f:t]
        info_batch = all_img_info[f:t]

        img_batch = []
        for el in path_batch:
            img_el = dict()
            for ch_name, ch_path in el.items():
                img_el[ch_name] = tif.imread(path_to_str(ch_path))
            img_batch.append(img_el)
        yield info_batch, img_batch


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


def main(method: str, dataset_dir: Path, batch_size: int):
    segm_channel_names = ("nucleus", "cell")
    out_base_img_name = "mask.ome.tiff"
    out_base_dir = Path("/output/")

    start = datetime.now()
    print('Started ' + str(start))

    segmenter = get_segmentation_method(method)
    img_batch_gen = load_img_batch(dataset_dir, segm_channel_names, batch_size)
    while True:
        try:
            info_batch, img_batch = next(img_batch_gen)
        except StopIteration:
            break
        print('Performing segmentation')
        segmented_batch = segmenter.segment(img_batch)
        print('Saving segmentation masks')
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
    args = parser.parse_args()

    main(args.method, args.dataset_dir, args.batch_size)
