import argparse
import os
import os.path as osp
from datetime import datetime
import faulthandler
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.python.client import device_lib

from batch import BatchLoader
from utils import path_to_str

Image = np.ndarray


def get_available_gpus() -> List[int]:
    # device_lib.list_local_devices() lists devices like this
    # [ PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
    #   PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
    def extract_gpu_id(gpu_name):
        return int(gpu_name.split(":")[-1])

    local_devices = device_lib.list_local_devices()
    gpu_ids = []
    for device in local_devices:
        if device.device_type == "GPU":
            gpu_ids.append(extract_gpu_id(device.name))
    if gpu_ids == []:
        raise ValueError("No GPUs were found")
    return gpu_ids


def get_allowed_gpu_ids(gpus: str) -> List[int]:
    found_gpus = get_available_gpus()
    print("Found GPU devices with ids:", found_gpus)
    if gpus == "":
        raise ValueError("No GPUs specified")
    elif gpus == "all":
        return found_gpus
    else:
        allowed_gpus = {int(_id) for _id in gpus.split(",") if _id != ""}
        gpu_ids = set(found_gpus) & allowed_gpus
        if not gpu_ids:
            msg = "Specified GPU ids: {allowed} do not match with found GPU ids: {found}"
            raise ValueError(msg.format(allowed=str(allowed_gpus), found=str(found_gpus)))
        return sorted(gpu_ids)


def remove_gpus_if_more_than_imgs(
    dataset_dir: Path, gpu_ids: List[int], segm_channels: Tuple[str]
):
    batcher = BatchLoader()
    batcher.dataset_dir = dataset_dir
    batcher.segmentation_channel_names = segm_channels
    num_imgs = batcher.get_num_of_imgs()

    updated_gpu_ids = gpu_ids
    if num_imgs > len(gpu_ids):
        updated_gpu_ids = gpu_ids[:num_imgs]
    return updated_gpu_ids


def run_segmentation(
    method: str, dataset_dir: Path, gpu_ids: List[int], segm_channels: Tuple[str]
):
    self_location = osp.realpath(osp.join(os.getcwd(), osp.dirname(__file__)))
    script_path = osp.join(self_location, "segment.py")
    cmd_template = (
        "CUDA_VISIBLE_DEVICES={gpu_id} "
        + ' python "{script_path}" '
        + " --method {method} "
        + ' --dataset_dir "{dataset_dir}" '
        + " --gpu_id {gpu_id} "
        + ' --gpus "{gpus}" '
        + ' --segm_channels "{segm_channels}" '
    )

    processes = []
    for gpu_id in gpu_ids:
        cmd = cmd_template.format(
            gpu_id=gpu_id,
            script_path=script_path,
            method=method,
            gpus=",".join(str(i) for i in gpu_ids),
            dataset_dir=path_to_str(dataset_dir),
            segm_channels=",".join(segm_channels),
        )
        processes.append(Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True))
    for proc in processes:
        returncode = proc.wait()
        print(proc.stdout.read())
        if returncode != 0:
            msg = "There was an error in the subprocess:\n " + proc.stderr.read()
            raise ChildProcessError(msg)


def main(method: str, dataset_dir: Path, gpus: str):
    start = datetime.now()
    # batch_size can't be larger for celldive + deepcell because all images have different shapes
    batch_size = 1
    segm_channels = ("nucleus", "cell")
    print("Started segmentation pipeline", str(start))
    print("Batch size is:", batch_size)
    gpus = gpus.lower()
    method = method.lower()

    gpu_ids = get_allowed_gpu_ids(gpus)
    gpu_ids = remove_gpus_if_more_than_imgs(dataset_dir, gpu_ids, segm_channels)
    run_segmentation(method, dataset_dir, gpu_ids, segm_channels)
    finish = datetime.now()
    print("Finished segmentation pipeline", str(finish))
    print("Time elapsed ", str(finish - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="segmentation method cellpose or deepcell")
    parser.add_argument("--dataset_dir", type=Path, help="path to directory with images")
    parser.add_argument(
        "--gpus", type=str, default="all", help="comma separated ids of gpus to use, e.g. 0,1,2"
    )
    parser.add_argument("--enable-faulthandler", action="store_true")
    args = parser.parse_args()
    if args.enable_faulthandler:
        print("Enabling Fault Handler")
        faulthandler.enable(all_threads=True)
    main(args.method, args.dataset_dir, args.gpus)
