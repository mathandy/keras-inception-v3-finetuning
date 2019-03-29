#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import cv2 as cv  # tested with version 3.4.1
import numpy as np


def overlay(image, mask, contrast=0.2, threshold=None):

    assert mask.ndim == 3 and mask.shape[2] == 3

    # normalize mask to interval [0, 1]
    if threshold is None:
        mask = mask/mask.max()
    else:
        mask = (mask > threshold) * 1
    return image*(1*mask + contrast*np.logical_not(mask))


def visualize(img, seg, polygons=None, gt_polygons=None, contrast=0.2):

    if seg.ndim == 3 and seg.shape[2] == 1:
        seg = np.stack(3*[seg[:, :, 0]], 2)
    elif seg.ndim == 2:
        seg = np.stack(3*[seg], 2)

    overlay_image = overlay(img, seg, contrast=contrast)

    if polygons is not None:
        for p in polygons:
            p = p.astype(int)
            overlay_image = cv.drawContours(overlay_image, [p], -1, (0, 255, 0), 1)
            seg = cv.drawContours(seg.astype('uint8'), [p], -1, (0, 255, 0), 1)
            img = cv.drawContours(img, [p], -1, (0, 255, 0), 1)
    if gt_polygons is not None:
        for p in gt_polygons:
            p = p.astype(int)
            overlay_image = cv.drawContours(overlay_image, [p], -1, (0, 0, 255), 1)
            seg = cv.drawContours(seg.astype('uint8'), [p], -1, (0, 0, 255), 1)
            img = cv.drawContours(img, [p], -1, (0, 0, 255), 1)

    if img.shape[0] < img.shape[1]:
        tiled = np.vstack((img, overlay_image, seg))
    else:
        tiled = np.hstack((img, overlay_image, seg))
    return tiled
