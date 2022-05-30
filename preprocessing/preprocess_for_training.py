# System
import argparse
import sys
from os import mkdir
from os.path import join,isdir
from tqdm import tqdm

# I/O
import imageio as io
import nibabel as nib

# Tools: Basic
import numpy as np

# Project Specific
from brainmets.utils import smart_ls, preproc_volume

def save_seg(path_save, vol):
    if not isdir(path_save):
        mkdir(path_save)

    new_vol = (vol * 255).astype(np.uint8)
    for ii in range(1, vol.shape[2] - 1):
        img = new_vol[:, :, ii]
        name_img = "{:03d}".format(ii) + '.png'
        io.imwrite(join(path_save, name_img), img)
    return 0

def save_volume(path_save, vol):
    if not isdir(path_save):
        mkdir(path_save)

    new_vol = (vol * 255).astype(np.uint8)
    for ii in range(1, vol.shape[2] - 1):
        img = new_vol[:, :, (ii - 1):(ii + 2)]
        name_img = "{:03d}".format(ii) + '.png'
        io.imwrite(join(path_save, name_img), img)
    return 0

def process_volume(path_pt, name_modality, imgsize, xys=None):
    path_nii = join(path_pt, name_modality + '.nii.gz')
    vol = nib.load(path_nii).get_fdata()
    vol,xys = preproc_volume(vol, imgsize, xys)
    return vol,xys

def process_and_save_data(opts):
    path_data = opts.path_data
    path_save = opts.path_save
    if not isdir(path_save):
        mkdir(path_save)

    for name_pt in tqdm(smart_ls(path_data)):
        path_data_pt = join(path_data, name_pt)
        path_save_pt = join(path_save, name_pt)
        if not isdir(path_save_pt):
            mkdir(path_save_pt)

        vol,xys = process_volume(path_data_pt, 'bravo', opts.imgsize)
        save_volume(join(path_save_pt, 'bravo'), vol)
        vol,_ = process_volume(path_data_pt, 'seg', opts.imgsize, xys)
        save_seg(join(path_save_pt, 'seg'), vol)


def main(args):
    parser = argparse.ArgumentParser(description="Preprocess Data.")

    # Paths
    parser.add_argument("--pData", dest="path_data", type=str, default=None)
    parser.add_argument("--pSave", dest="path_save", type=str, default=None)

    # Hyperparameters
    parser.add_argument("--imgsize", dest="imgsize", type=int, default=512)

    # Creating Object
    opts = parser.parse_args(args[1:])
    process_and_save_data(opts)

    return 0

if __name__ == '__main__':
    main(sys.argv)