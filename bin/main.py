import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from utils import alpha_num_order, make_dir_if_not_exists
import deepcell_wrapper
import cellpose_wrapper


def run_segmentation(modality: str, img_dir: Path, out_dir: Path):
    out_dir_deepcell = out_dir / "deepcell"
    out_dir_cellpose = out_dir / "cellpose"
    make_dir_if_not_exists(out_dir_deepcell)
    make_dir_if_not_exists(out_dir_cellpose)

    deepcell_wrapper.main(
        img_dir=img_dir,
        out_dir=out_dir_deepcell,
    )

    cellpose_wrapper.main(
        img_dir=img_dir,
        out_dir=out_dir_cellpose,
    )


#
# def check_if_segm_imgs_in_dir(dir_path: Path):
#     names = ('nucleus', 'membrane', 'cytoplasm')
#     listing = list(dir_path.iterdir())
#     if listing == []:
#         return listing
#     else:
#         found_channels = dict()
#         for path in listing:
#             path_str = path_to_str(path)
#             found = {name: path_str for name in names if name in path_str}
#             if found != {}:
#                 found_channels.update(found)
#         if len(found_channels.keys()) == len(names):
#             return True
#         else:
#             print('Not all channels found in', dir_path,
#                   'Found channels', found_channels)
#             return False


def collect_img_dirs(dataset_dir: Path):
    img_dirs = [p for p in list(dataset_dir.iterdir()) if p.is_dir()]
    img_dirs = sorted(img_dirs, key=lambda path: alpha_num_order(path.name))
    img_dirs_dict = dict()
    for img_dir in img_dirs:
        img_dirs_dict[img_dir.name] = img_dir
    return img_dirs_dict


def main(modality: str, dataset_dir: Path):
    img_dirs = collect_img_dirs(dataset_dir)
    out_dir = Path("/output/")
    for dir_name, dir_path in img_dirs.items():
        img_out_dir = out_dir / dir_name
        run_segmentation(modality, dir_path, img_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, help="mibi or celldive")
    parser.add_argument("--dataset_dir", type=Path, help="path to directory with images")
    args = parser.parse_args()

    main(args.modality, args.dataset_dir)
