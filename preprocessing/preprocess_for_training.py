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

def process_volume(path_pt, name_modality, imgsize):
    path_nii = join(path_pt, name_modality + '.nii.gz')
    vol = nib.load(path_nii).get_fdata()
    vol = preproc_volume(vol, imgsize=imgsize)
    return vol

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
        if opts.bool_bravo:
            vol = process_volume(path_data_pt, 'bravo', opts.imgsize)
            save_volume(join(path_save_pt, 'bravo'), vol)
        if opts.bool_flair:
            vol = process_volume(path_data_pt, 'flair', opts.imgsize)
            save_volume(join(path_save_pt, 'flair'), vol)
        if opts.bool_t1_gd:
            vol = process_volume(path_data_pt, 't1_gd', opts.imgsize)
            save_volume(join(path_save_pt, 't1_gd'), vol)
        if opts.bool_t1_pre:
            vol = process_volume(path_data_pt, 't1_pre', opts.imgsize)
            save_volume(join(path_save_pt, 't1_pre'), vol)
        if opts.bool_seg:
            vol = process_volume(path_data_pt, 'seg', opts.imgsize)
            save_seg(join(path_save_pt, 'seg'), vol)

def main(args):
    parser = argparse.ArgumentParser(description="Preprocess Data.")

    # Paths
    parser.add_argument("--pData", dest="path_data", type=str, default=None)
    parser.add_argument("--pSave", dest="path_save", type=str, default=None)

    # Hyperparameters
    parser.add_argument("--imgsize", dest="imgsize", type=int, default=512)

    # Switches
    parser.add_argument("--bBravo", dest="bool_bravo", type=int, default=0)
    parser.add_argument("--bFlair", dest="bool_flair", type=int, default=0)
    parser.add_argument("--bT1_gd", dest="bool_t1_gd", type=int, default=0)
    parser.add_argument("--bT1_pre", dest="bool_t1_pre", type=int, default=0)
    parser.add_argument("--bSeg", dest="bool_seg", type=int, default=0)

    # Creating Object
    opts = parser.parse_args(args[1:])
    process_and_save_data(opts)

    return 0

if __name__ == '__main__':
    main(sys.argv)