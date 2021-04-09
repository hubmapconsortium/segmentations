import argparse
import os
import sys
from pathlib import Path
from subprocess import PIPE, run

sys.path.append(os.getcwd())
from utils import make_dir_if_not_exists, path_to_str


def main(modality: str, img_dir: Path):
    cmd_template = "python {wrapper} --img_dir {img_dir} --out_dir {out_dir}"

    out_dir = Path("./output")
    out_dir_deepcell = out_dir / "deepcell"
    out_dir_cellpose = out_dir / "cellpose"
    make_dir_if_not_exists(out_dir_deepcell)
    make_dir_if_not_exists(out_dir_cellpose)

    cmd1 = cmd_template.format(
        wrapper="deepcell_wrapper.py",
        img_dir=path_to_str(img_dir),
        out_dir=path_to_str(out_dir_deepcell),
    )

    cmd2 = cmd_template.format(
        wrapper="cellpose_wrapper.py",
        img_dir=path_to_str(img_dir),
        out_dir=path_to_str(out_dir_cellpose),
    )

    run(cmd1, shell=True, stdout=PIPE, stderr=PIPE)
    run(cmd2, shell=True, stdout=PIPE, stderr=PIPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, help="mibi or celldive")
    parser.add_argument("--img_dir", type=Path, help="path to directory with images")
    args = parser.parse_args()

    main(args.modality, args.img_dir)
