# utils.py>
import os
import re
from os.path import join
from pathlib import Path

import numpy as np
import tifffile as tif
from skimage.segmentation import find_boundaries


def alpha_num_order(string: str) -> str:
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alpha_num_order("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join(
        [format(int(x), "05d") if x.isdigit() else x for x in re.split(r"(\d+)", string)]
    )


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def get_boundary(mask):
    mask_boundary = find_boundaries(mask)
    mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
    return mask_boundary_indexed


def fill_in_ome_meta_template(size_y: int, size_x: int, dtype) -> str:
    template = """<?xml version="1.0" encoding="utf-8"?>
            <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
              <Image ID="Image:0" Name="mask.ome.tiff">
                <Pixels BigEndian="true" DimensionOrder="XYZCT" ID="Pixels:0" SizeC="4" SizeT="1" SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" Type="{dtype}">
                    <Channel ID="Channel:0:0" Name="cells" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:1" Name="nuclei" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:2" Name="cell_boundaries" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:3" Name="nucleus_boundaries" SamplesPerPixel="1" />
                    <TiffData FirstC="0" FirstT="0" FirstZ="0" IFD="0" PlaneCount="1" />
                    <TiffData FirstC="1" FirstT="0" FirstZ="0" IFD="1" PlaneCount="1" />
                    <TiffData FirstC="2" FirstT="0" FirstZ="0" IFD="2" PlaneCount="1" />
                    <TiffData FirstC="3" FirstT="0" FirstZ="0" IFD="3" PlaneCount="1" />
                </Pixels>
              </Image>
            </OME>
        """
    ome_meta = template.format(size_y=size_y, size_x=size_x, dtype=np.dtype(dtype).name)
    return ome_meta


def save_segmentation_masks(out_dir, cell, nuc, cell_b, nuc_b):
    dtype = np.uint32
    ome_meta = fill_in_ome_meta_template(cell.shape[0], cell.shape[1], dtype)
    out_path = path_to_str(out_dir / "mask.ome.tiff")
    TF = tif.TiffFile(out_path)

    TF.write(cell.astype(dtype), contiguous=True, photometric="minisblack", description=ome_meta)
    TF.write(nuc.astype(dtype), contiguous=True, photometric="minisblack", description=ome_meta)
    TF.write(cell_b.astype(dtype), contiguous=True, photometric="minisblack", description=ome_meta)
    TF.write(nuc_b.astype(dtype), contiguous=True, photometric="minisblack", description=ome_meta)

    TF.close()
