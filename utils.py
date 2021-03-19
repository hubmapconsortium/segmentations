# utils.py>
import os
from os.path import join
from skimage.segmentation import find_boundaries
import numpy as np
from tifffile import TiffWriter

def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_boundary(mask):
	mask_boundary = find_boundaries(mask)
	mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
	return mask_boundary_indexed

def save_ome_tiff(img_dir, cell, nuc, cell_b, nuc_b):
	img_name = 'segmentation.ome.tiff'
	cell = cell.astype(np.int32)
	nuc = nuc.astype(np.int32)
	cell_b = cell_b.astype(np.int32)
	nuc_b = nuc_b.astype(np.int32)
	with TiffWriter(join(img_dir, img_name)) as tif:
		tif.save(cell)
		tif.save(nuc)
		tif.save(cell_b)
		tif.save(nuc_b)
		metadata={'axes': 'ZYX', 'Plane': {'PositionZ': [0.0, 1.0, 2.0, 3.0]}}
