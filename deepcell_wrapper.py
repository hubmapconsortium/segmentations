from skimage.io import imread
from deepcell.applications import MultiplexSegmentation
import os
from os.path import join
import numpy as np
import sys
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
sys.path.append(os.getcwd())
from utils import *

if __name__ == '__main__':
	
	# read nucleus and membrane channels
	file_dir = sys.argv[1]
	im1 = imread(join(file_dir, 'nucleus.tif'))
	im2 = imread(join(file_dir, 'membrane.tif'))
	im = np.stack((im1, im2), axis=-1)
	im = np.expand_dims(im, 0)
	
	# setup tensorflow
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
	
	# predict the masks of nucleus and the whole cell
	app = MultiplexSegmentation(use_pretrained_weights=True)
	labeled_image = app.predict(im, compartment='both')
	labeled_image = np.squeeze(labeled_image, axis=0)
	cell_mask = labeled_image[:, :, 0]
	nuc_mask = labeled_image[:, :, 1]
	
	# get the mask boundaries
	cell_boundary_mask = get_boundary(cell_mask)
	nuc_boundary_mask = get_boundary(nuc_mask)
	save_dir = join(file_dir, 'deepcell')
	os.makedirs(save_dir)
	save_ome_tiff(save_dir, cell_mask, nuc_mask, cell_boundary_mask, nuc_boundary_mask)
