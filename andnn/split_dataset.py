""" Split a dataset into a train/val/test subsets.

Usage
-----
* Typical usage::

    $ python split_dataset.py unsplit_dir split_dir 5 5


* To remove any files encountered that can't be read with `cv.imread()`,
use the `-u` or `--remove_unreadable` flag::

    $ python split_dataset.py -u unsplit_dir split_dir 5 5


"""


import os
import cv2 as cv
import numpy as np
from shutil import copytree


def make_subset(split_dir, subset, n_subset, remove_unreadable=False):
    train_dir = os.path.join(split_dir, 'train')
    subset_dir = os.path.join(split_dir, subset)

    # setup `subset_dir`
    os.mkdir(subset_dir)
    for category in os.listdir(train_dir):
        os.mkdir(os.path.join(subset_dir, category))

    for category in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, category)):
            continue

        # remove unreadable images
        if remove_unreadable:
            images = [f for f in os.listdir(os.path.join(train_dir, category))]
            for fn in images:
                img = cv.imread(os.path.join(train_dir, category, fn))
                try:
                    img.shape
                except AttributeError:
                    os.remove(os.path.join(train_dir, category, fn))

        # move `n_subset` images of this subset to
        images = [f for f in os.listdir(os.path.join(train_dir, category))]
        for fn in np.random.choice(images, n_subset, replace=False):
            os.rename(os.path.join(train_dir, category, fn),
                      os.path.join(subset_dir, category, fn))


def split_dataset(data_dir, out_dir, n_val, n_test, remove_unreadable=False):

    # copy dataset to training dir
    os.mkdir(out_dir)
    copytree(data_dir, os.path.join(out_dir, 'train'))

    if n_val:
        make_subset(out_dir, 'val', n_val, remove_unreadable)
    if n_test:
        make_subset(out_dir, 'test', n_test, remove_unreadable)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        help="Root dir of original unsplit dataset.  Assumed to be "
             "divided into subdirectories by class.")
    parser.add_argument(
        "out_dir",
        help="Where to store new split dataset.")
    parser.add_argument(
        "n_val", type=int,
        help="The number of images per class to put in the validation set.")
    parser.add_argument(
        "n_test", type=int,
        help="The number of images per class to put in the validation set.")
    parser.add_argument(
        '-u', '--remove_unreadable', default=False, action='store_true',
        help="Don't include any unreadable images encountered.")
    args = parser.parse_args()

    split_dataset(data_dir=args.data_dir,
                  out_dir=args.out_dir,
                  n_val=args.n_val,
                  n_test=args.n_test,
                  remove_unreadable=args.remove_unreadable)
