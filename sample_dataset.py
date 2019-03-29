import os
from shutil import copyfile
import numpy as np


def is_jpeg_or_png(fn):
    return os.path.splitext(fn)[1][1:].lower() in ('jpg', 'jpeg', 'png')


def sample_dataset(data_dir, out_dir, n, assert_out_dir_does_not_exist=False):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        if assert_out_dir_does_not_exist:
            raise

    all_samples = []
    for directory, _, files in os.walk(data_dir):
        for fn in files:
            if not is_jpeg_or_png(fn):
                continue
            full_path = os.path.join(directory, fn)
            all_samples.append(full_path)

    for full_path in np.random.choice(all_samples, n, replace=False):
        copyfile(full_path,
                 os.path.join(out_dir, os.path.basename(full_path)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        help="data directory with subdirectory labels")
    parser.add_argument(
        "out_dir",
        help="where to store sample")
    parser.add_argument(
        "num_samples", type=int,
        help="how many samples to collect")
    parser.add_argument(
        "-a", "--add_samples", default=False, action='store_true',
        help="Specify it's ok if `out_dir` already exists, add images.")
    args = parser.parse_args()

    sample_dataset(data_dir=args.data_dir,
                   out_dir=args.out_dir,
                   n=args.num_samples,
                   assert_out_dir_does_not_exist= not args.add_samples)
