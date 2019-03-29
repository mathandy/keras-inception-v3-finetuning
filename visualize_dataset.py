"""Create grid/sample of images from dataset with subdirectory labels."""


import os
import cv2 as cv
import numpy as np


def is_image(filename):
    x = cv.imread(filename)
    try:
        x.shape
    except AttributeError:
        return False
    return True


def is_JPEG_or_PNG(filename, extensions=('jpg', 'jpeg', 'png')):
    return os.path.splitext(filename)[-1][1:].lower() in extensions


def rescale_by_height(image, target_height, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv.resize(image, (w, target_height), interpolation=method)


def rescale_by_width(image, target_width, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv.resize(image, (target_width, h), interpolation=method)


def add_caption(image, text, font=cv.FONT_HERSHEY_PLAIN, font_size=8,
                thickness=4, color=(0, 0, 0), background_color=(255, 255, 255),
                margin=5, rotate=0):
    image = np.rot90(image, rotate)
    # create background for text
    (txt_w, txt_h), line_height = \
        cv.getTextSize(text, font, font_size, thickness)
    h, w = image.shape[:2]
    caption = [[background_color] * w] * (line_height + txt_h + margin)
    caption = np.array(caption, dtype='uint8')

    bl = w - margin - txt_w, margin + txt_h + line_height
    cv.putText(caption, text, bl, font, font_size, color, thickness,
               lineType=cv.LINE_AA)
    return np.rot90(np.vstack((image, caption)), (4 - rotate) % 4)


def stack_images(images, scale=1, out_name=None, vertical=True, captions=None,
                 side_captions=False):

    if all(isinstance(x, str) for x in images):
        images = [cv.imread(f) for f in images]

    if captions:
        images = [add_caption(x, s, rotate=int(side_captions))
                  for x, s in zip(images, captions)]

    widest_image = images[np.argmax([x.shape[1] for x in images])]
    target_w = cv.resize(widest_image, (0, 0), fx=scale, fy=scale).shape[1]

    if vertical:
        stacked_images = \
            np.vstack((rescale_by_width(x, target_w) for x in images))
    else:
        stacked_images = \
            np.hstack((rescale_by_height(img, target_w) for img in images))
    if out_name:
        cv.imwrite(out_name, stacked_images)
    return stacked_images


def visualize_dataset(image_dir, n, replace=False, out_name=None, scale=1.0):
    subdirs = [d for d in os.listdir(image_dir)
               if os.path.isdir(os.path.join(image_dir, d))]

    class_samples = []
    for subdir in subdirs:
        images = [os.path.join(image_dir, subdir, fn)
                  for fn in os.listdir(os.path.join(image_dir, subdir))
                  if is_image(os.path.join(image_dir, subdir, fn))]
        class_sample = np.random.choice(images, n, replace=replace)
        class_samples.append(stack_images(class_sample, scale, vertical=False))
    sample = stack_images(class_samples, captions=subdirs, side_captions=True)
    if out_name:
        cv.imwrite(out_name, sample)
    return sample


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_dir",
        help="Image directory ~ must contain a subdir for each class.")
    parser.add_argument(
        "-n", "--n_sample", default=5, type=int,
        help="number of samples from each class")
    parser.add_argument(
        "-s", "--scale", default=1.0, type=float,
        help="number of samples from each class")
    parser.add_argument(
        '-r', "--with_replacement", default=False, action='store_true',
        help="Sample w/ replacement.")
    parser.add_argument(
        '-o', "--out_name", default=None,
        help="Where to save the output image.")
    args = parser.parse_args()

    visualize_dataset(image_dir=args.image_dir,
                      n=args.n_sample,
                      replace=args.with_replacement,
                      out_name=args.out_name,
                      scale=args.scale)
