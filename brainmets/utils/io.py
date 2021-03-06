from os import listdir

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes

def smart_ls(path):
    list_items = listdir(path)
    list_items = [item for item in list_items if item[0] != '.']
    return sorted(list_items)

def normalize_volume(vol):
    # Volumes go from inferior to superior.
    vol = vol.astype(np.float32)
    vol -= vol.min()
    vol /= vol.max() + 1e-6
    return vol

def fill_holes_volume(vol_bw):
    for ii in range(vol_bw.shape[2]):
        vol_bw[:,:,ii] = binary_fill_holes(vol_bw[:,:,ii])
    return vol_bw

#def resize_volume(vol, imgsize=512):
#    vol_new = np.zeros((imgsize,imgsize,vol.shape[2]))
#    for ii in range(vol.shape[2]):
#        vol_new[:,:,ii] = cv2.resize(vol, (imgsize,imgsize))
#    vol_new = normalize_volume(vol)
#    return vol_new

def preproc_volume(vol, imgsize=512, xys=None):
    vol = normalize_volume(vol)
    if xys is None:
        thresh = threshold_otsu(vol)
        vol_bw = vol > thresh
        #vol_bw = fill_holes_volume(vol_bw)
        ys, xs, _ = np.where(vol_bw)
        ymin = ys.min()
        ymax = ys.max() + 1
        xmin = xs.min()
        xmax = xs.max() + 1
    else:
        ymin,ymax,xmin,xmax = xys
    vol = vol[ymin:ymax, xmin:xmax]
    vol = cv2.resize(vol, (imgsize,imgsize))
    vol = normalize_volume(vol)
    return vol, (ymin,ymax,xmin,xmax)

