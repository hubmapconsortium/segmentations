import os
from os.path import join

from skimage.io import imread, imsave


def get_MIBI_slices(img_dir):
    nucleus_channel = "cp " + img_dir + "/TIFs/HH3.tif " + img_dir + "/nucleus.tif"
    cyto_channel = "cp " + img_dir + "/TIFs/PanKeratin.tif " + img_dir + "/cytoplasm.tif"
    membrane_channel = "cp " + img_dir + "/TIFs/ECadherin.tif " + img_dir + "/membrane.tif"
    os.system(nucleus_channel)
    os.system(cyto_channel)
    os.system(membrane_channel)


def get_CellDIVE_slices(img_dir, img_name):
    img_path = join(img_dir, img_name)
    image = imread(img_path)
    nucleus = 0
    cytoplasm = 2
    membrane = 15
    imsave(join(img_dir, "nucleus.tif"), image[nucleus, :])
    imsave(join(img_dir, "cytoplasm.tif"), image[cytoplasm, :])
    imsave(join(img_dir, "membrane.tif"), image[membrane, :])
