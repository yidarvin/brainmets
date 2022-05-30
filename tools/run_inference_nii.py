# System
import argparse
import sys

# Tools: Basic
import cv2
import numpy as np
import nibabel as nib

# Tools: ML
import torch
import torch.nn.functional as F

# Project Specific
from brainmets.nets import deeplabv3_50
from brainmets.utils import preproc_volume,normalize_volume

def run_inference(model, img, vol, xys):
    H,W,_ = vol.shape
    out = np.zeros((H,W))
    X = img.transpose(2,0,1)
    X = X.astype(np.float32)
    X -= X.min()
    X /= X.max() + 1e-6
    X = torch.from_numpy(X).float().unsqueeze(0).cuda()
    with torch.set_grad_enabled(False):
        output = model(X)
        prob = F.softmax(output,dim=1)
    prob = prob.detach().cpu().numpy()[0,1,:,:]

    ymin,ymax,xmin,xmax = xys
    prob = cv2.resize(prob, (xmax-xmin,ymax-ymin))
    out[ymin:ymax, xmin:xmax] = prob
    return out

def main(args):
    parser = argparse.ArgumentParser(description="Run Inference.")

    # Paths
    parser.add_argument("--pModel", dest="path_model", type=str, default=None)
    parser.add_argument("--pNii", dest="path_nii", type=str, default=None)

    # Hyperparameters
    parser.add_argument("--imgsize", dest="imgsize", type=int, default=512)

    # Creating Object
    opts = parser.parse_args(args[1:])
    path_model = opts.path_model
    path_nii = opts.path_nii
    path_save = path_nii + '.out.nii.gz'

    # loading in the model
    model = deeplabv3_50(in_chan=3, out_chan=2, pretrained=True)
    model = model.cuda()
    model.load_state_dict(torch.load(path_model))
    model.eval()

    # creating the volume
    vol = nib.load(path_nii).get_fdata()
    vol = normalize_volume(vol)
    vol_new, xys = preproc_volume(vol)

    # looping through the volume
    out = np.zeros_like(vol)
    for ii in range(1, vol.shape[2] - 1):
        img = vol_new[:, :, (ii - 1):(ii + 2)]
        out[:, :, ii] = run_inference(model, img, vol, xys)

    # Saving niifty output
    img = nib.Nifti1Image(out, np.eye(4))
    nib.save(img, path_save)

    return 0

if __name__ == '__main__':
    main(sys.argv)
