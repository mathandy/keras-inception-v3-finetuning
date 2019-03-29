#!/usr/bin/env python
"""

Usage
-----
Typical usage::

    $ python test_keras_classifier.py <root_data_dir> <model.h5> \
        <image_width> <image_height> -r <results_dir>

Orientation example::

    $ python test_keras_classifier.py fish-orientation-data/ \
        fish-orientation-model.h5 299 299 -r orientation_test_results


"""

from __future__ import print_function
from keras.models import load_model
import os
import cv2 as cv
import numpy as np


def get_sample_filenames(sample_dir):
    samples = []
    for _, _, files in os.walk(sample_dir):
        for fn in files:
            samples.append(os.path.join(sample_dir, fn))
    return samples


def make_predictions(samples, model, dsize, results_dir=None):
    predictions = []
    for fn in samples:
        resized_sample = cv.resize(cv.imread(fn), dsize=dsize)
        probabilities = model.predict(np.expand_dims(resized_sample, 0))
        predicted_label = probabilities.argmax()
        predictions.append(predicted_label)

        out_path = os.path.join(results_dir, str(predicted_label) + '.txt')
        with open(out_path, 'a+') as out:
            out.write('"' + os.path.abspath(fn) + '"\n')
    return predictions


def score(model, dsize, class_data_dir, results_dir=None):
    class_name = os.path.split(class_data_dir)[-1]
    num_classes = model.output_shape[-1]

    filenames = get_sample_filenames(class_data_dir)
    predictions = make_predictions(filenames, model, dsize, results_dir)

    print('%s: (n = %s)' % (class_name, len(filenames)))
    counts = dict(zip(*np.unique(predictions, return_counts=True)))
    for index in range(num_classes):
        print('\t%s: %s' % (index, counts.get(index, 0)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        help="data directory with subdirectory labels")
    parser.add_argument(
        "model_path",
        help="keras model to test")
    parser.add_argument(
        "image_width", type=int,
        help="width to resize images to")
    parser.add_argument(
        "image_height", type=int,
        help="height to resize images to")
    parser.add_argument(
        '-r', "--results_dir",
        help="Where to store filenames group by predictions.")
    args = parser.parse_args()

    # DATA_DIR = 'fish-orientation-data'
    # MODEL_PATH = 'fish-orientation-model.h5'
    # IMG_DSIZE = (299, 299)
    
    image_dims = (args.image_width, args.image_height)

    # get subdirectories of data dir
    class_dirs = [os.path.join(args.data_dir, subdir) 
                  for subdir in os.listdir(args.data_dir)]
    class_dirs = [x for x in class_dirs if os.path.isdir(x)]

    # prep results directory
    if args.results_dir is not None and not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # load model and score
    m = load_model(args.model_path)
    for class_dir in class_dirs:

        # prep class results directory
        class_results_dir = None
        if args.results_dir:
            class_results_dir = \
                os.path.join(args.results_dir, os.path.split(class_dir)[-1])
            if not os.path.exists(class_results_dir):
                os.mkdir(class_results_dir)

        # run samples through model and report results
        score(m, image_dims, class_dir, results_dir=class_results_dir)
