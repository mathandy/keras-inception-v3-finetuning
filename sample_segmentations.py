#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Review image segmentations.

This script makes it fast and easy to review and hand-score image
segmentations.  A text file, "reviewed.txt" will be created and appended
to with your scores as you go.  Further instructions will be provided
when the script is run.

Requirements
------------
* Numpy
* OpenCV (tested with version 3.4.1)

Recommended Usage
-----------------
Call this script from a directory containing an two subdirectories named
"images" and "segmentations" as well as (if relevant) the dictionary
of affine transforms, "affine_transforms.npz", created in the
post-processing steps (see `segmentation_postprocessing.py`).::

    $ cd path/to/parent_dir_of_images_and_segmentations
    $ python path/to/visualize.py


For more options::

    $ python visualize.py -h


"""


from __future__ import print_function, division
import os
from glob import glob
import cv2 as cv  # tested with version 3.4.1
import numpy as np


def is_JPEG_or_PNG(filename, extensions=('jpg', 'jpeg', 'png')):
    return os.path.splitext(filename)[-1][1:].lower() in extensions


def scale_with_padding(image, dsize, pad_with=0):
    if image.ndim == 3 and isinstance(pad_with, int):
        pad_with = (pad_with, pad_with, pad_with)
    ssize = image.shape[:2][::-1]
    x_diff, y_diff = np.array(ssize) - dsize
    assert not (x_diff % 2) and not (y_diff % 2)
    new_image = cv.resize(image, dsize)

    rows = [[pad_with]*dsize[0]]*(x_diff//2)
    new_image = np.vstack((rows, new_image, rows))
    cols = np.hstack([[[pad_with]]*ssize[1]]*(y_diff//2))
    new_image = np.hstack((cols, new_image, cols))
    return new_image


def overlay(image, mask, contrast=0.2):
    if mask.ndim == 2:
        mask = np.stack((mask, mask, mask), 2)
    mask = (mask > 0) * 1

    if mask.shape[0] > image.shape[0]:
        mask = mask[:image.shape[0]]

    return (image*(1 * mask + contrast * (np.logical_not(mask)))).astype('uint8')


def get_sample_pair_by_name(name, multiple_file_warning=False,
                            return_filenames=False):
    images = glob(os.path.join('images', name) + '.*')
    segmentation_masks = glob(os.path.join('segmentations', name) + '.*')
    if multiple_file_warning and len(images) > 1:
        print("Warning: multiple images begin with %s" % name)
        for image in images:
            print(image)
    elif not images:
        raise IOError("No image found beginning with name %s" % name)
    if multiple_file_warning and len(segmentation_masks) > 1:
        print("Warning: multiple segmentation masks begin with %s" % name)
        for image in images:
            print(image)
    elif not segmentation_masks:
        raise IOError("No segmentation mask found beginning with name %s" % name)
    if return_filenames:
        return images[0], segmentation_masks[0]
    return cv.imread(images[0]), cv.imread(segmentation_masks[0])


def affine_inv(transform):
    # y = ax + b
    # x = a^(-1)*(y - b)
    A_inv, b = np.linalg.inv(transform[:, :2]), transform[:, 2]
    return np.hstack((A_inv, -np.dot(A_inv, b).reshape(2, 1)))


def visualize(name=None, thickness=1, crop_size=None, transform=None,
              scale_dsize=None, contrast=0.2, polygons=None, gt_polygons=None,
              mode='default', fit_to_crop=False, debug=False):
    if name is None:
        name = np.random.choice(os.listdir('images'))[:-4]
    img, seg = get_sample_pair_by_name(name, True)

    if debug:
        from ipdb import set_trace
        set_trace()

    if scale_dsize is not None:
        seg = scale_with_padding(seg, scale_dsize)

    if transform is not None:
        overlay_seg = cv.warpAffine(seg, affine_inv(transform),
                                    img.shape[:2][::-1])
        overlay_img = img
    else:
        overlay_img, overlay_seg = img, seg
    # seg = 1 * (seg[:img.shape[0]] > 0)
    overlay_image = overlay(overlay_img, overlay_seg, contrast=contrast)

    if seg.ndim == 2:
        seg = np.stack([seg]*3, 2)
    # seg = seg * 255

    if polygons is not None:
        for p in polygons:
            p = p.astype(int)
            overlay_image = cv.drawContours(overlay_image, [p], -1, (255, 0, 0),
                                            thickness=thickness)
            seg = cv.drawContours(seg.astype('uint8'), [p], -1, (255, 0, 0),
                                  thickness=thickness)
            img = cv.drawContours(img, [p], -1, (255, 0, 0),
                                  thickness=thickness)
    if gt_polygons is not None:
        for p in gt_polygons:
            p = p.astype(int)
            overlay_image = cv.drawContours(overlay_image, [p], -1, (0, 0, 255),
                                            thickness=thickness)
            seg = cv.drawContours(seg.astype('uint8'), [p], -1, (0, 0, 255),
                                  thickness=thickness)
            img = cv.drawContours(img, [p], -1, (0, 0, 255),
                                  thickness=thickness)

    # # make three pane image showing original, overlay, and segmentation
    # if img.shape[0] < img.shape[1]:
    #     tiled = np.vstack((img, overlay_image, seg))
    # else:
    #     tiled = np.hstack((img, overlay_image, seg))

    if mode == 'overlay':
        image2return = overlay_image
    elif mode == 'segmentation':
        image2return = seg
    elif mode == 'default':
        image2return = img
    else:
        raise ValueError('mode = %s not understood.' % mode)

    if crop_size is not None:
        assert len(gt_polygons) == 1 and len(polygons) == 1
        crop_size = np.array(crop_size)

        x_min, y_min = np.concatenate((polygons, gt_polygons)).min(axis=(0, 1))
        x_max, y_max = np.concatenate((polygons, gt_polygons)).max(axis=(0, 1))
        center = ((np.array([x_max, y_max]) + [x_min, y_min])/2).astype(int)
        w, h = crop_size

        if fit_to_crop:
            j0, i0 = int(np.floor(x_min)), int(np.floor(y_min))
            j1, i1 = int(np.ceil(x_max)), int(np.ceil(y_max))
            image2return = cv.resize(image2return[i0:i1, j0:j1], dsize=crop_size)
        else:
            j0, i0 = center - crop_size//2
            j0, i0 = max(j0, 0), max(i0, 0)
            j1, i1 = j0 + w, i0 + h
            image2return = image2return[i0:i1, j0:j1]
    return image2return


def sample(root_dir, size, thickness, crop=True, output_path='sample.jpg',
           transforms_path='affine_transforms.npz',
           polygons_path=None, gt_polygons_path=None, mode='default',
           skip_true_negatives=False):

    transforms = dict()
    try:
        transforms = np.load(transforms_path)
    except:
        print("\nProblem loading transforms_path = %s.  If segmentations "
              "don't need to be transformed to match images, this is ok."
              "\n" % transforms_path)

    # load polygons (if provided)
    polygons = dict()
    if polygons_path is not None:
        polygons = np.load(polygons_path)
    gt_polygons = dict()
    if gt_polygons_path is not None:
        gt_polygons = np.load(gt_polygons_path)

    crop_size = None
    if crop and (polygons or gt_polygons):
        all_polygons = np.concatenate(
            [x for x in polygons.values() if len(x)] +
            [x for x in gt_polygons.values() if len(x)])
        crop_size = np.array([p.max(axis=0) - p.min(axis=0)
                              for p in all_polygons]).max(axis=0)
        crop_size = np.ceil(crop_size).astype(int)

    # change working directory to `root_dir`
    _working_dir = os.getcwd()
    if root_dir is None:
        root_dir = _working_dir
    os.chdir(root_dir)

    # get sample of names
    image_names = [os.path.splitext(f)[0] for f in os.listdir('images')
                   if is_JPEG_or_PNG(f)]
    if skip_true_negatives:
        image_names = [n for n in image_names
                       if len(polygons[n]) and len(gt_polygons[n])]
        print("%s positive samples found." % len(image_names))
    image_sample = np.random.choice(image_names, np.product(size), replace=False)

    # get results for sample
    sampled_results = [visualize(name=name,
                                 thickness=thickness,
                                 crop_size=crop_size,
                                 transform=transforms.get(name, None),
                                 polygons=polygons.get(name, None),
                                 gt_polygons=gt_polygons.get(name, None))
                       for name in image_sample]
    os.chdir(_working_dir)

    # make grid from results
    w, h = size
    grid = np.array(sampled_results)
    grid = grid.reshape((h, w) + grid.shape[1:])
    grid = np.vstack((np.hstack((s for s in grid[i])) for i in range(h)))

    # output
    cv.imwrite(output_path, grid)
    return grid


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("-r", "--root_dir", default=os.getcwd(),
                      help="directory containing 'images' and "
                           "'segmentations' subdirectories")
    args.add_argument("--size", nargs=2, type=int, default=(4, 3),
                      help="Width and height respectively of sample grid.")
    args.add_argument("--thickness", type=int, default=1,
                      help="Thickness of drawn polygon/spline prediction.")
    args.add_argument("-o", "--output_path", default='sample.jpg',
                      help="where to store the output sample.")
    args.add_argument("-p", "--polygons_path",
                      default=None,
                      help="path to npz storing ground truth polygons to draw")
    args.add_argument("-g", "--gt_polygons_path",
                      default=None,
                      help="path to npz storing predicted polygons to draw")
    args.add_argument("-t", "--transforms_path",
                      default='affine_transforms.npz',
                      help="affine transforms npz file")
    args.add_argument("-m", "--mode",
                      default='default',
                      help="'overlay', 'segmentation', or 'default'")
    args.add_argument("--include_true_negatives", default=False,
                      action='store_true',
                      help="Use this to allow true negatives in sample.")
    args = vars(args.parse_args())

    # check for polygons in `root_dir`
    polygons_path = args['polygons_path']
    gt_polygons_path = args['gt_polygons_path']
    if polygons_path is None:
        default_path = os.path.join(args['root_dir'], 'predicted_polygons.npz')
        if os.path.exists(default_path):
            polygons_path = default_path
    if gt_polygons_path is None:
        default_path = os.path.join(args['root_dir'], 'gt_polygons.npz')
        if os.path.exists(default_path):
            gt_polygons_path = default_path
    if polygons_path and gt_polygons_path:
        print("\nGround truth is red, predictions are blue.\n")

    # create sample
    tp = args['transforms_path']
    if tp.lower() == 'none':
        tp = None
    sample(root_dir=args['root_dir'],
           size=args['size'],
           thickness=args['thickness'],
           output_path=args['output_path'],
           transforms_path=tp,
           polygons_path=polygons_path,
           gt_polygons_path=gt_polygons_path,
           mode=args['mode'],
           skip_true_negatives=not args['include_true_negatives'])
