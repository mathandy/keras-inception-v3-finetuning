from __future__ import division, absolute_import, print_function

import os
import random

import numpy as np
from scipy.misc import imresize
from sklearn.decomposition import IncrementalPCA
from skimage.transform import resize
from imageio import imread
import cv2 as cv
# from tflearn.data_utils import build_hdf5_image_dataset

from andnn.utils import Timer, is_image


exists = os.path.exists
join = os.path.join


def k21hot(Y, k=None):
    """Convert integer labels to 1-hot labels."""
    if k is None:
        k = len(np.unique(Y.ravel()))  # number of classes
    Yflat = Y.ravel()
    hot_labels = np.zeros(Yflat.shape + (k,), dtype=Y.dtype)
    hot_labels[np.arange(Yflat.shape[0]), Yflat.astype(int) % k] = 1
    return hot_labels.reshape(Y.shape + (k,))


def k2pixelwise(Y, im_shape, k=None, onehot=False):
    """Convert integer labels to pixel-wise labels."""
    if k is None:
        k = len(np.unique(Y))  # number of classes
    pw = np.tensordot((Y % k).ravel(), np.ones(im_shape), axes=0)
    if onehot:
        return k21hot(pw, k)
    else:
        return pw


def shuffle_together(list_of_arrays, permutation=None):
    m = len(list_of_arrays[0])
    assert all([len(x) == m for x in list_of_arrays[1:]])

    if permutation is None:
        permutation = list(range(m))
        random.shuffle(permutation)
    return [x[permutation] for x in list_of_arrays], permutation


def split_data(X, Y, validpart=0, testpart=0, shuffle=False):
    """Split data into training, validation, and test sets.  

    Assumes examples are indexed by first dimension.

    Args:
        X: any sliceable iterable
        Y: any sliceable iterable
        validpart: int or float proportion
        testpart: int or float proportion
        shuffle: bool

    Returns:
        (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    """
    assert validpart or testpart

    m = len(X)

    # shuffle data
    if shuffle:
        (X, Y), permutation = shuffle_together((X, Y))

    if 0 < validpart < 1 or 0 < testpart < 1:
        m_valid = int(validpart * m)
        m_test = int(testpart * m)
        m_train = len(Y) - m_valid - m_test
    else:
        m_valid = validpart
        m_test = testpart
        m_train = m - m_valid - m_test

    X_train = X[:m_train]
    Y_train = Y[:m_train]

    X_valid = X[m_train: m_train + m_valid]
    Y_valid = Y[m_train: m_train + m_valid]

    X_test = X[m_train + m_valid: len(X)]
    Y_test = Y[m_train + m_valid: len(Y)]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


# def load_h5(root_image_dir, image_shape, h5_dataset_name=None, mode='folder',
#             categorical_labels=True, normalize=True, grayscale=False,
#             files_extensions=('.png', '.jpg', '.jpeg'), chunks=False,
#             ignore_existing=False):
#     if h5_dataset_name is None:
#         h5_dataset_name = os.path.join(root_image_dir,
#                                        os.path.split(root_image_dir)[
#                                            -1] + '.h5')
#     if mode == 'unlabeled':
#         unlabeled = True
#     else:
#         unlabeled = False
#     if ignore_existing or not os.path.exists(h5_dataset_name):
#         print("Creating hdf5 file...", end='')
#         if unlabeled:
#             mode = 'file'
#             # create txt file listing images
#             imlist_filename = os.path.join(root_image_dir, 'imlist.txt')
#             with open(imlist_filename, 'w') as imlist:
#                 ims = os.listdir(root_image_dir)
#                 for im in ims:
#                     line = os.path.join(root_image_dir, im) + ' 0\n'
#                     imlist.write(line)
#                 root_image_dir = imlist_filename
#
#         build_hdf5_image_dataset(root_image_dir,
#                                  image_shape=image_shape,
#                                  output_path=h5_dataset_name,
#                                  mode=mode,
#                                  categorical_labels=categorical_labels,
#                                  normalize=normalize,
#                                  grayscale=grayscale,
#                                  files_extension=files_extensions,
#                                  chunks=chunks)
#         print('Done.')
#
#     h5f = h5py.File(h5_dataset_name, 'r')
#     X, Y = h5f['X'], h5f['Y']
#     if unlabeled:
#         return X
#     return X, Y


# def whiten(X):
#     """PCA whitening."""
#     X -= np.mean(X, axis=0)  # zero-center
#     cov = np.dot(X.T, X) / X.shape[0]  # compute the covariance matrix
#     U, S, V = np.linalg.svd(cov)
#     X = np.dot(X, U)  # decorrelate the data
#     X /= np.sqrt(S + 1e-5)  # divide by the eigenvalues
#     return X


def incremental_whiten(X):
    def _whiten(A):
        ipca = IncrementalPCA(n_components=A.shape[1], whiten=True)
        return ipca.fit_transform(A)

    # center
    X -= np.mean(X, axis=0)

    # split channels and flatten
    m, w, h, d = X.shape
    assert m >= w * h
    channels = np.moveaxis(X, 3, 0).reshape(d, m, w * h)

    # whiten
    whitened_channels = np.stack([_whiten(c) for c in channels])
    # import pdb; pdb.set_trace()

    # put channels back in original shape and return
    return np.moveaxis(whitened_channels.reshape(d, m, w, h), 0, 3)


def image_preloader(image_directory, size, image_depth=3, label_type=False,
                    pixel_labels_lookup=None, num_classes=None,
                    exts=('.jpg', '.jpeg', '.png'), normalize=True,
                    shuffle=True, onehot=True, testpart=0, validpart=0,
                    whiten=False, ignore_existing=False,
                    storage_directory=None,
                    save_split_sets=True, seed=None):
    """Image pre-loader (for use when entire dataset can be loaded in memory).

    This function is designed to load images store in directories in one of the
    following fashions (as determined by `label_type`):
        - subdirectory_labels images are in `image_directory` and 
        - 
        -

    Args:
        image_directory (string): The root directory containing all images.
        size (array-like): The (width, height) images should be re-sized to.
        image_depth: the number of color channels in images.
        label_type (string): see description above.
        pixel_labels_lookup (func): function that returns an image given the
            filename and shape of an image.
        num_classes (int): Number of label classes.  Required only if 
            `label_type`=="pixel".
        exts (iterable): The set of acceptable file extensions
        normalize: 
        shuffle: 
        onehot: 
        testpart: 
        validpart: 
        whiten: 
        ignore_existing: 
        storage_directory (string):  Should be provided if numpy arrays are to
            be saved (to save time if training images are ever be loaded again).
        save_split_sets (bool):

    Returns:
        (X_train, Y_train, X_valid, Y_valid, X_test, Y_test, class_names)
    """
    exts = [ext.lower() for ext in exts]

    if storage_directory is None:
        from warnings import warn
        warn("no storage_directory provided -- unable to save/load.")
    elif not os.path.exists(storage_directory):
        os.mkdir(storage_directory)
        # raise IOError("storage_directory='{}' does not exist."
        #               "".format(storage_directory))
    
    _label_types = ['subdirectory', 'pixel', None]
    # some parameter-checking
    if label_type == 'subdirectory':
        pass
    elif label_type == 'pixel':
        if pixel_labels_lookup is None:
            raise ValueError("If label_type=pixel, then pixel_labels_lookup "
                             "must be provided.")
        if num_classes is None:
            raise ValueError("If label_type=pixel, then num_classes "
                             "must be provided.")
    elif label_type is None:
        pass
    else:
        mes = "`label_type` must be one of the following options\n"
        mes += '\n'.join(_label_types)
        raise ValueError(mes)

    # some useful definitions
    s = '-{}x{}.npy'.format(*size)

    if image_depth in [None, 0, 1]:
        shape_of_input_images = size
    else:
        shape_of_input_images = (size[0], size[1], image_depth)

    if label_type == "pixel":
        shape_of_pixel_labels = (size[0], size[1], num_classes)

    def is_image(image_file_name):
        return os.path.splitext(image_file_name)[1].lower() in exts

    if label_type == "subdirectory":
        class_names = [d for d in os.listdir(image_directory)
                       if os.path.isdir(join(image_directory, d))]
    else:
        class_names = None

    if storage_directory is not None:
        Yfile = join(storage_directory, 'Ydata' + s)
        Xfile = join(storage_directory, 'Xdata' + s)
        Xfile_white = join(storage_directory, 'Xdata-whitened-' + s)

    _no_npy_found=False
    if not (storage_directory is None or ignore_existing):

        if exists(Yfile) and \
              (exists(Xfile) or (whiten and exists(Xfile_white))):
            with Timer("loading target data from .npy files"):
                Y = np.load(Yfile)
            if whiten:
                if os.path.exists(Xfile_white):
                    with Timer("loading whitened data from .npy files"):
                        X = np.load(Xfile_white)
                else:
                    with Timer("loading unwhitened data from .npy files"):
                        X = np.load(Xfile)

                    with Timer('Whitening'):
                        X = incremental_whiten(X)

                    with Timer('Saving whitened data'):
                        np.save(Xfile_white, X)
            else:
                with Timer("loading input data from .npy file"):
                    X = np.load(Xfile)
        else:
            mes = 'No numpy file found.'
            _no_npy_found=True
    else:
        mes = ''
        _no_npy_found=True
        
    if _no_npy_found:
        with Timer(mes + 'Loading data from image directories'):

            with Timer('Collecting image file names'):
                if label_type == "subdirectory":
                    image_files = []
                    Y = []
                    for k, d in enumerate(class_names):
                        with Timer(d):
                            fd = join(image_directory, d)
                            image_files_d = [join(fd, fn)
                                             for fn in os.listdir(fd)
                                             if is_image(fn)]
                            Y += [k] * len(image_files_d)
                            image_files += image_files_d
                elif label_type == "pixel":
                    image_files = [join(image_directory, fn)
                                   for fn in os.listdir(image_directory)
                                   if is_image(fn)]
                else:  # Y is filenames
                    Y = image_files = [join(image_directory, fn)
                                       for fn in os.listdir(image_directory)
                                       if is_image(fn)]

            with Timer('\tLoading/Resizing images'):

                # initialize arrays
                X = np.empty((len(image_files), size[0], size[1], image_depth),
                             dtype=np.float32)
                if label_type == "subdirectory":
                    Y = np.array(Y).astype(np.float32)
                elif label_type == "pixel":
                    Y = np.empty(
                        (X.shape[0], X.shape[1], X.shape[2], num_classes),
                        dtype=np.float32)

                # resize and load images into arrays
                if label_type == "pixel":
                    for k, fn in enumerate(image_files):
                        imx = imread(fn).astype(np.float32)
                        imy = pixel_labels_lookup(fn, imx.shape)
                        if imx.shape != shape_of_input_images:
                            X[k] = imresize(imx, shape_of_input_images)
                            Y[k] = np.stack([imresize(imy[:, :, l], size)
                                             for l in range(num_classes)],
                                            axis=2)
                        else:
                            X[k] = imx
                            Y[k] = imy
                else:
                    for k, fn in enumerate(image_files):
                        im = imread(fn).astype(np.float32)
                        if im.shape != shape_of_input_images:
                            X[k, :, :, :] = imresize(im, shape_of_input_images)
                        else:
                            X[k, :, :, :] = im

        if shuffle:
            if seed is None:
                from time import time
                seed = int(time())
            print('shuffling dataset with seed = %s' % seed)
            with Timer('Shuffling'):
                m = len(Y)

                permutation = np.random.RandomState(seed=seed).permutation(m)
                random.shuffle(permutation)
                X = X[permutation]
                Y = Y[permutation]

        if onehot and label_type == "subdirectory":
            with Timer('Converting to 1-hot labels'):
                Y = k21hot(Y)

        if storage_directory is not None:
            with Timer('Saving data (before any normalizing, whitening, and/or '
                       'splitting)'):
                np.save(Xfile, X)
                np.save(Yfile, Y)

        if whiten:
            with Timer('Whitening'):
                X = incremental_whiten(X)

            if storage_directory is not None:
                with Timer('Saving whitening data'):
                    np.save(Xfile_white, X)

    if normalize and not whiten:
        with Timer('Normalizing data'):
            # X = (X - np.mean(X))/np.std(X)
            X -= np.mean(X, axis=0)
            X /= np.std(X, axis=0)

    if (testpart or validpart):
        with Timer('Splitting data into fit/validate/test sets'):
            X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
                split_data(X, Y, validpart, testpart, shuffle=False)

        if save_split_sets:

            with Timer('Saving split datasets'):
                np.save(join(storage_directory, 'Xtrain' + s), X_train)
                np.save(join(storage_directory, 'Ytrain' + s), Y_train)
                if validpart:
                    np.save(join(storage_directory, 'Xvalid' + s), X_valid)
                    np.save(join(storage_directory, 'Yvalid' + s), Y_valid)
                if testpart:
                    np.save(join(storage_directory, 'Xtest' + s), X_test)
                    np.save(join(storage_directory, 'Ytest' + s), Y_test)
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, class_names
    else:
        return X, Y, None, None, None, None, class_names


def load_npys(npy_dir, samples_per_class, image_shape=(28, 28), flat=True):
    """Load dataset stored as directory of '.npy' files, one each class."""
    spc = samples_per_class
    npys = [fn for fn in os.listdir(npy_dir) if fn.endswith('.npy')]
    num_classes = len(npys)

    # get x-data
    if flat:
        x = np.empty((spc*num_classes, np.product(image_shape)), dtype='uint8')
    else:
        x = np.empty((spc*num_classes,) + image_shape, dtype='uint8')
    class_names = []
    for k, npy in enumerate(npys):
        x[k*spc: (k+1)*spc] = np.load(os.path.join(npy_dir, npy))[:spc]
        class_names.append(os.path.splitext(npy)[0])

    # get y-data
    y = []
    for k, npy in enumerate(npys):
        yk = [False] * num_classes
        yk[k] = True
        y += [yk] * spc
    y = np.array(y, dtype='bool')
    if flat:
        x = x.reshape((-1,) + image_shape)
    return x, y, class_names


def load_split_npy_data(npy_dir, samples_per_class, testpart, valpart=0,
                        resize=False, cast=False, seed=314159,
                        image_shape=(28, 28), flat=True):
    assert 0 <= testpart + valpart <= 1
    x, y, class_names = load_npys(npy_dir, samples_per_class,
                                  image_shape=image_shape, flat=flat)
    p = np.random.RandomState(seed=seed).permutation(len(x))
    x, y = x[p], y[p]

    if resize:
        x_new = np.empty(shape=[len(x)] + list(resize[: x.ndim - 1]),
                         dtype=cast if cast else x.dtype)
        dsize = resize[:2][::-1]
        for k in range(len(x)):
            x_new[k] = cv.resize(x[k], dsize=dsize)
        x = x_new
        if len(resize) == 3 and x.ndim == 3:  # convert to color images
            assert resize[2] == 3
            x = np.stack([x] * 3, axis=x.ndim)

    if cast:
        x = x.astype(cast)
        y = y.astype(cast)

    n_test = int(len(x) * testpart)
    n_val = int(len(x) * valpart)

    def _split(arr):
        test = arr[:n_test]
        val = arr[n_test: n_test + n_val]
        train = arr[n_test + n_val:]
        return train, val, test

    x_train, x_val, x_test = _split(x)
    y_train, y_val, y_test = _split(y)
    return x_train, y_train, x_val, y_val, x_test, y_test, class_names


def save_images_to_npys(image_dir, image_size, output_dir,
                        problem_images_filename='problem-images.txt'):
    image_size = tuple(image_size)  # dsize
    npy_shapes = []
    problem_images = []
    for subdir in os.listdir(image_dir):
        subdir_full = os.path.join(image_dir, subdir)
        if not os.path.isdir(subdir_full):
            continue
        with Timer('Working on %s' % subdir):
            image_fns = [fn for fn in os.listdir(subdir_full) if is_image(fn)]
            # images = np.empty((len(image_fns),) + image_size[::-1] + (3,),
            #                   dtype='uint8')
            images = []
            for fn in image_fns:
                fn_full = os.path.join(subdir_full, fn)
                try:
                    image = resize(imread(fn_full), image_size + (3,))
                except:
                    print('Failed to load/resize %s' % fn_full)
                    problem_images.append(fn_full)
                    pass
                images.append(image)
            np.save(os.path.join(output_dir, subdir + '.npy'), images)
            npy_shapes.append(np.array(images).shape)
    for x in npy_shapes:
        print(x)
    with open(problem_images_filename, 'w+') as f:
        for x in problem_images:
            f.write(x + '\n')
    print('problem images output to "%s"' % problem_images_filename)