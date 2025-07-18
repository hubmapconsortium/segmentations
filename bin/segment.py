import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from batch import BatchLoader
from img_proc.match_masks import get_matched_masks
from utils import path_to_str, write_stack_to_file

Image = np.ndarray


def save_masks(
    base_out_dir: Path,
    base_img_name: str,
    info: List[Dict[str, str]],
    imgs: List[Dict[str, Image]],
    allow_cells_only = False
):
    # HACK
    # TODO: clean this up properly
    if not info and len(imgs) == 1:
        info = [{".": ""}]
    for i, el in enumerate(info):
        # TODO: figure out why we only want the first
        dir_name, img_set = list(el.items())[0]
        out_dir = base_out_dir / dir_name
        out_dir.mkdir(exist_ok=True, parents=True)
        img_prefix = f"{img_set}_" if img_set else ""
        img_name = img_prefix + base_img_name
        channels = imgs[i]
        mask_stack = np.stack(
            [
                channels["cell"],
                channels["nucleus"],
                channels["cell_boundary"],
                channels["nucleus_boundary"],
            ],
            axis=0,
        )
        st = datetime.now()
        #write_stack_to_file(path_to_str(out_dir / "test"), mask_stack,0)
        print("Started matching cell and nuclei", str(st))
        matched_stack, fraction_matched = get_matched_masks(mask_stack, True, allow_cells_only=allow_cells_only)
        fin = datetime.now()
        print("Finished matching", str(fin))
        print("Time elapsed", str(fin - st))
        img_out_path = path_to_str(out_dir / img_name)
        write_stack_to_file(img_out_path, matched_stack, round(fraction_matched, 3))


def get_segmentation_method(method: str):
    if method == "cellpose":
        from cellpose_wrapper import CellposeWrapper

        segmenter = CellposeWrapper()
    elif method == "deepcell":
        from deepcell_wrapper import DeepcellWrapper

        segmenter = DeepcellWrapper()
    else:
        msg = "Incorrect segmentation method " + method
        raise ValueError(msg)
    print("Using segmentation method ", method)
    return segmenter


def main(method: str, dataset_dir: Path, gpu_id: str, gpu_ids: str, segm_channels_str: str, allow_cells_only=False):
    out_base_img_name = "mask.ome.tiff"
    out_base_dir = Path("/output/")

    gpu_id = int(gpu_id)
    gpu_ids = [int(i) for i in gpu_ids.split(",")]
    segm_channels = segm_channels_str.split(",")

    batcher = BatchLoader()
    batcher.dataset_dir = dataset_dir
    batcher.segmentation_channel_names = segm_channels
    batcher.gpu_ids = gpu_ids
    batcher.init_img_batch_generator_per_gpu()

    img_batch_gen = batcher.get_batch_gen_for_gpu(gpu_id)
    segmenter = get_segmentation_method(method)

    while True:
        try:
            info_batch, img_batch = next(img_batch_gen)
        except StopIteration:
            break
        segmented_batch = segmenter.segment(img_batch)
        save_masks(out_base_dir, out_base_img_name, info_batch, segmented_batch, allow_cells_only=allow_cells_only)
    segmenter.close_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="segmentation method cellpose or deepcell")
    parser.add_argument("--dataset_dir", type=Path, help="path to directory with images")
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")
    parser.add_argument(
        "--gpus", type=str, default="0", help="comma separated ids of gpus to use, e.g. 0,1,2"
    )
    parser.add_argument("--allow_cells_only", type=bool, default=False, help="allow cells without nucleus")
    parser.add_argument(
        "--segm_channels",
        type=str,
        default="nucleus,cell",
        help="comma separated types of channels for segmentation",
    )
    args = parser.parse_args()

    main(args.method, args.dataset_dir, args.gpu_id, args.gpus, args.segm_channels, allow_cells_only=args.allow_cells_only)
