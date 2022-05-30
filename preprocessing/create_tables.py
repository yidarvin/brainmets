# System
import argparse
import sys
import random
import csv
from os.path import join,isdir
from tqdm import tqdm

# I/O
import imageio as io

# Project Specific
from brainmets.utils import smart_ls

with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

def write_table(path_save_csv, path_data, list_pts):
    with open(path_save_csv, 'w', newline='') as csvfile:
        fieldnames = ['img', 'seg', 'nonempty']
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)
        for name_pt in tqdm(list_pts):
            path_pt = join(path_data, name_pt)
            path_bravo = join(path_pt, 'bravo')
            path_seg = join(path_pt, 'seg')
            for name_img in smart_ls(path_bravo):
                img = io.imread(join(path_seg, name_img))
                nonempty = 0
                if (img > 100).sum() > 0:
                    nonempty = 1
                row = [join(path_bravo, name_img), join(path_seg, name_img), str(nonempty)]
                writer.writerow(row)


def make_tables(opts):
    path_data = opts.path_data
    list_pts = smart_ls(path_data)
    list_pts = [item for item in list_pts if isdir(join(path_data,item))]
    random.shuffle(list_pts)

    list_pts_val = list_pts[:opts.valsize]
    list_pts_test = list_pts[opts.valsize:(opts.valsize+opts.testsize)]
    list_pts_train = list_pts[(opts.valsize+opts.testsize):]

    write_table(join(path_data, 'val.csv'), path_data, list_pts_val)
    write_table(join(path_data, 'test.csv'), path_data, list_pts_test)
    write_table(join(path_data, 'train.csv'), path_data, list_pts_train)

    return 0

def main(args):
    parser = argparse.ArgumentParser(description="Create Tables.")

    # Paths
    parser.add_argument("--pData", dest="path_data", type=str, default=None)

    # Hyperparameters
    parser.add_argument("--valsize", dest="valsize", type=int, default=10)
    parser.add_argument("--testsize", dest="testsize", type=int, default=0)

    # Creating Object
    opts = parser.parse_args(args[1:])
    make_tables(opts)

    return 0

if __name__ == '__main__':
    main(sys.argv)