from __future__ import absolute_import, division, print_function
import tensorflow as tf
from sys import stdout
import os
from time import time as current_time
from PIL import Image, ImageDraw
from imageio import imread, imwrite
from skimage.transform import resize 
from skimage.viewer import ImageViewer
import numpy as np
import cv2 as cv


def rescale_by_height(image, target_height, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv.resize(image, (w, target_height), interpolation=method)


def rescale_by_width(image, target_width, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv.resize(image, (target_width, h), interpolation=method)


def is_readable_image(filename):
    x = cv.imread(filename)
    try:
        x.shape
    except AttributeError:
        return False
    return True


def is_image(fn, extensions=('jpg', 'jpeg', 'png')):
    return os.path.splitext(fn)[1][1:].lower() in extensions


def is_jpeg_or_png(fn):
    return os.path.splitext(fn)[1][1:].lower() in ('jpg', 'jpeg', 'png')


def resize_images(images, max_dim=None, img_shape=None,
                  interpolation=cv.INTER_LANCZOS4):
    assert max_dim or img_shape and not (max_dim and img_shape)
    resized_images = np.empty(shape=(images.shape[0],) + tuple(img_shape),
                              dtype=images.dtype)

    if img_shape:
        dsize = img_shape[:2][::-1]
        for k, image in enumerate(images):
            resized_images[k] = cv.resize(image, dsize=dsize,
                                          interpolation=interpolation)
    else:
        for k, image in enumerate(images):
            h, w = image.shape[:2]
            if h > w:
                resized_images[k] = \
                    rescale_by_height(image, max_dim, interpolation)
            else:
                resized_images[k] = \
                    rescale_by_width(image, max_dim, interpolation)

    # convert to color images
    if len(img_shape) == 3 and resized_images.ndim == 3:
        assert img_shape[2] == 3
        return np.stack([resized_images] * 3, axis=resized_images.ndim)
    return resized_images


def show_image_sample(image_directory, grid_shape, extensions=('jpg',),
                      show=True, output_filename=None, thumbnail_size=100,
                      assert_enough_images=True):
    """Creates image of a grid of images sampled from `image_directory`."""

    if isinstance(thumbnail_size, int):
        size = (thumbnail_size, thumbnail_size)
    else:
        size = thumbnail_size
    
    image_files = [os.path.join(image_directory, fn)
                   for fn in os.listdir(image_directory)
                   if is_image(fn, extensions)]

    assert len(grid_shape) == 2
    if assert_enough_images:
        assert grid_shape[0]*grid_shape[1] <= len(image_files)

    sample = np.random.choice(image_files, grid_shape)

    grid = []
    for i in range(grid_shape[0]):
        row = []
        for j in range(grid_shape[1]):
            row.append(resize(imread(sample[i, j]), size))
        grid.append(np.concatenate(row, axis=1))
    grid = (255*np.concatenate(grid)).astype('uint8')

    if show:
        ImageViewer(grid).show()

    if output_filename is not None:
        imwrite(output_filename, grid)


def softmax(z):
    ez = np.exp(z - z.max())  # prevents blow-up, suggested in cs231n
    return ez/ez.sum()


def pnumber(x, n=5, pad=' '):
    """Takes in a float, outputs a string of length n."""
    s = str(x)
    try:
        return s[:n]
    except IndexError:
        return pad*(n - len(s)) + s


def ppercent(x, n=5, pad=' '):
    """Takes in a float, outputs a string (percentage) of length n."""
    return pnumber(x, n=n - 1, pad=pad) + '%'


class Timer:
    """A simple tool for timing code while keeping it pretty."""

    def __init__(self, mes='', pretty_time=True, n=4, pad=' ', enable=True):
        self.mes = mes  # append after `mes` + '...'
        self.pretty_time = pretty_time
        self.n = n
        self.pad = pad
        self.enabled = enable

    def format_time(self, et, n=4, pad=' '):
        if self.pretty_time:
            if et < 60:
                return '{} sec'.format(pnumber(et, n, pad))
            elif et < 3600:
                return '{} min'.format(pnumber(et / 60, n, pad))
            else:
                return '{} hrs'.format(pnumber(et / 3600, n, pad))
        else:
            return '{} sec'.format(et)

    def __enter__(self):
        if self.enabled:
            stdout.write(self.mes + '...')
            stdout.flush()
            self.t0 = current_time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = current_time()
        if self.enabled:
            print("done (in {})".format(
                self.format_time(self.t1 - self.t0, self.n, self.pad)))
            stdout.flush()


def find_nth(big_string, substring, n):
    """Find the nth occurrence of a substring in a string."""
    idx = big_string.find(substring)
    while idx >= 0 and n > 1:
        idx = big_string.find(substring, idx + len(substring))
        n -= 1
    return idx


def parentdir(path_, n=1):
    for i in range(n):
        path_ = os.path.dirname(path_)
    return path_


class WorkingDirectory:
    """ A tool to temporarily change the current working directory, then change 
    it back upon `__exit__` (i.e. for use with `with`).

    Usage:
        with WorkingDirectory(directory_to_work_in):
            # do something
    """

    def __init__(self, working_directory):
        self.old_wd = os.getcwd()
        self.wd = working_directory

    def __enter__(self):
        os.chdir(self.wd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.old_wd)


# class BoxedImage:
#     def __init__(self, textboxes, image_filename, gt_filename):
#         self.textboxes = textboxes
#         self.image_filename = image_filename
#         self.gt_filename = gt_filename
#
#     def __repr__(self):
#         s = "from: " + self.gt_filename + '\n'
#         s += "image: " + self.image_filename
#         for tb in self.textboxes:
#             s += '\n' + str(tb)
#         return s


class TextBox:
    def __init__(self, coords, text, image_filename=None, gt_filename=None):
        self.coords = coords
        self.text = text
        self.image_filename = image_filename
        self.gt_filename = gt_filename

    def __repr__(self):
        s = "from: " + self.gt_filename
        s += "image: " + self.image_filename
        return str(self.coords) + ' : ' + self.text


def draw_boxes_on_image(image, textboxes, savename=None, show=True):
    """
    Draw boxes on an image.

    Args:

        images (string): The filename of an image. 
        textboxes (iterable): A list of TextBox objects.
        savename (object): If none, won't save.
        show (bool): Whether or not to display immediately.
    """
    with Image.open(image) as img:
        draw = ImageDraw.Draw(img)

        def z2xy(z):
            return z.real, z.imag

        for tb in textboxes:
            for i in range(4):
                x1, y1 = z2xy(tb.coords[i])
                x2, y2 = z2xy(tb.coords[(i + 1) % 4])
                draw.line((x1, y1, x2, y2), fill=128)
        del draw
        if savename is not None:
            img.create_checkpoint(savename)
    return img


# def color_in_polygon(points, im, color=1):
#     from svgpathtools import Path, Line, path_encloses_pt
#     p = Path(*[Line(points[i], points[(i + 1) % len(points)])
#                for i in range(len(points))])
#
#     x0 = min(z.real for z in points)
#     x1 = max(z.real for z in points)
#     y0 = min(z.imag for z in points)
#     y1 = max(z.imag for z in points)
#
#     for y in range(y0, y1+1):
#         for x in range(x0, x1+1):
#             if path_encloses_pt(p, x+1j*y):
#                 im[y, x] = 1


def color_in_polygon(img, points, color=255):
    """Colors in polygon in image (in place).

    Args:
        points: a list of (x,y) points
        img: a numpy.array
        color: 3-tuple or integer 0-255

    Returns:
        None
    """
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv.fillConvexPoly(img, pts, True, color)


def boxes2silhouette(boxes, size, dtype=np.float32):
    """Creates a binary image displaying the boxes.

    Args:
        boxes: A list of 4-tuples of complex numbers.
        size: a 2-tuple or 3-tuple, the output image's size.

    Returns:
        numpy.array: A numpy.array of specified `dtype` -- though all values 
        are either 0 or 1.
    """

    def z2xy(z):
        return [(z.real, z.imag) for z in z]

    sil = np.zeros(size, dtype=np.uint8)
    for box in boxes:
        color_in_polygon(sil, z2xy(box), color=1)

    return np.array(sil).astype(dtype=dtype)


def boxes2pixellabels(boxes, size, dtype=np.float32):
    """Creates a 3-tensor (numpy.array) of shape `[size[0], size[1], 2]`.

    Note:  This is meant for (the 1-hot analog) of binary pixel labels.

    Args:
        boxes: A list of 4-tuples of complex numbers.
        size: a 2-tuple or 3-tuple, the output image's size.

    Returns:
        numpy.array: A numpy.array of specified `dtype` and dimensions
        `[size[0], size[1], 2]` -- though all values are either 0 or 1.
    """

    sil = boxes2silhouette(boxes=boxes, size=size, dtype=dtype)
    return np.stack([sil, 1 - sil], axis=2)


def step_plot(list_of_lists, ylabels=None):
    import matplotlib.pyplot as plt
    from pylab import subplot, subplots_adjust

    m = len(list_of_lists[0])
    assert all(len(v) == m for v in list_of_lists)
    steps = range(m)

    subplots_adjust(hspace=0.000)
    number_of_subplots=3

    for i, y in enumerate(list_of_lists):
        ax = subplot(number_of_subplots, 1, i+1)
        ax.plot(steps, y)
        ax.set_xlabel('step')

        if ylabels is None:
            ax.set_ylabel('unnamed_%s'%i)
        else:
            ax.set_ylabel(ylabels[i])
        # ax.set_title('Simple XY point plot')
    plt.show()


def accuracy(predictions, labels):
    is_correct = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(is_correct, "float"))


def num_correct(predictions, labels):
    is_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(is_correct, "float"))


def num_incorrect(predictions, labels):
    is_incorrect = tf.not_equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(is_incorrect, "float"))


def batches(data, batch_size, include_remainder=True):
    """Break a dataset, `data` into batches of size `batch_size`.

    If `len(data) % batch_size > 0`, the remaining
    examples will be included if `include_remainder` is true."""
    num_batches = data.shape[0] // batch_size
    if len(data) % batch_size and include_remainder:
        num_batches += 1
    return (data[k * batch_size: min((k + 1) * batch_size, len(data))]
            for k in range(num_batches))


def color_image(image, num_classes=20):
    # taken from fcn.utils
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))
