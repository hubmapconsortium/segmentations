import os
import sys
sys.path.append(os.getcwd())
from input_channels import *

if __name__ == '__main__':
	modality = sys.argv[1]
	img_dir = sys.argv[2]
	if modality == 'MIBI':
		get_MIBI_slices(img_dir)
	elif modality == 'CellDIVE':
		img_name = sys.argv[3]
		get_CellDIVE_slices(img_dir, img_name)
	os.system('python deepcell_wrapper.py ' + img_dir)
	os.system('python cellpose_wrapper.py ' + img_dir)
