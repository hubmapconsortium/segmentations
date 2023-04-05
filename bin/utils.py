import re
from pathlib import Path
from typing import List

import numpy as np
import tifffile as tif
from skimage.segmentation import find_boundaries

Image = np.ndarray


def alpha_num_order(string: str) -> str:
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alpha_num_order("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join(
        [format(int(x), "05d") if x.isdigit() else x for x in re.split(r"(\d+)", string)]
    )


def alpha_num_order_filename(path: Path) -> str:
    return alpha_num_order(path.name)


def path_to_str(path: Path):
    return str(path.absolute().as_posix())


def make_dir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_img_listing(in_dir: Path) -> List[Path]:
    allowed_extensions = (".tif", ".tiff")
    listing = list(in_dir.iterdir())
    img_listing = [f for f in listing if f.suffix in allowed_extensions]
    img_listing = sorted(img_listing, key=lambda x: alpha_num_order(x.name))
    return img_listing


def get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def get_boundary(masks: List[Image]) -> List[Image]:
    boundaries = []
    for mask in masks:
        mask_boundary = find_boundaries(mask, mode="inner")
        mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
        boundaries.append(mask_boundary_indexed)
    return boundaries


def fill_in_ome_meta_template(size_y: int, size_x: int, dtype, match_fraction: float) -> str:
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
              <StructuredAnnotations>
                <XMLAnnotation ID="Annotation:0">
                    <Value>
                        <OriginalMetadata>
                            <Key>FractionOfMatchedCellsAndNuclei</Key>
                            <Value>{match_fraction}</Value>
                        </OriginalMetadata>
                    </Value>
                </XMLAnnotation>
              </StructuredAnnotations>
            </OME>
        """
    ome_meta = template.format(
        size_y=size_y, size_x=size_x, dtype=np.dtype(dtype).name, match_fraction=match_fraction
    )
    return ome_meta


def write_stack_to_file(out_path: str, stack, mismatch: float):
    dtype = np.uint32
    ome_meta = fill_in_ome_meta_template(stack.shape[-2], stack.shape[-1], dtype, mismatch)
    stack_shape = stack.shape
    new_stack_shape = [stack_shape[0], 1, stack_shape[1], stack_shape[2]]
    with tif.TiffWriter(out_path, bigtiff=True) as TW:
        TW.write(
            stack.reshape(new_stack_shape).astype(dtype),
            contiguous=True,
            photometric="minisblack",
            description=ome_meta,
        )
