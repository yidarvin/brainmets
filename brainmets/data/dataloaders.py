import csv
import cv2
import imageio as io
import numpy as np

from torch.utils.data import Dataset,DataLoader



class BravoOnly(Dataset):
    def __init__(self, path_csv, resize=512, bloat=0, inference=False, transform=None):
        self.path_csv = path_csv
        self.resize = resize
        self.transform = transform
        self.inference = inference

        self.path_data = []
        with open(self.path_csv) as csvfile:
            pathreader = csv.reader(csvfile, delimiter=",")
            for ii,row in enumerate(pathreader):
                if ii == 0:
                    continue
                if self.inference:
                    self.path_data.append((row[0] ,None))
                else:
                    if int(row[2])==1:
                        self.path_data += [(row[0], row[1]) for jj in range(bloat)]
                    self.path_data.append((row[0], row[1]))
    def __len__(self):
        return len(self.path_data)
    def __getitem__(self, idx):
        path_img, path_seg = self.path_data[idx]
        # Reading in Images and Processing
        img = io.imread(path_img).astype(np.float32)
        img -= img.min()
        img /= img.max() + 1e-6
        if path_seg is not None:
            seg = io.imread(path_seg)
            if len(seg.shape) == 3:
                seg = seg[:,:,0]
            seg = (seg > 100) + 0
        else:
            seg = None
        # Resizing
        resize = self.resize
        if resize != img.shape[0] or resize != img.shape[1]:
            img = cv2.resize(img, (resize ,resize))
            img -= img.min()
            img /= img.max() + 1e-6
        if seg is not None and (resize != seg.shape[0] or resize != seg.shape[1]):
            seg = cv2.resize(seg.astype(np.float32), (resize, resize), interpolation=cv2.INTER_NEAREST)
        img = img.transpose(2 ,0 ,1)
        sample = {"X" :img, "Y" :seg}
        if self.transform:
            sample = self.transform(sample)

        return sample