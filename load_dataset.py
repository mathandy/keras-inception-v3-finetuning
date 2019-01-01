import tensorflow as tf
import numpy as np
import os
import cv2 as cv
from skimage.io import imread
from skimage.transform import resize

try:
    from .andnn_util import Timer
except:
    from andnn_util import Timer


def is_image(fn, extensions=('jpg', 'jpeg', 'png')):
    return os.path.splitext(fn)[1][1:].lower() in extensions


def load_npys(npy_dir, samples_per_class, image_shape=(28, 28), flat=True):
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


def load_split_data(npy_dir, samples_per_class, testpart, valpart=0,
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


# def npys_to_split_tfdataset(testpart, valpart=0, npy_dir=NPY_DIR,
#                             samples_per_class=SAMPLES_PER_CLASS, seed=314159):
#
#     x_train, y_train, x_val, y_val, x_test, y_test = \
#     data = dict(zip(
#         ('x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test'),
#         load_split_data(testpart, valpart, npy_dir, samples_per_class, seed)
#     ))
#
#
#     dataset = tf.contrib.data.Dataset.from_tensor_slices(actual_data)
#     iterator = dataset.make_initializable_iterator()
#
#     sess.run(iterator.initializer, feed_dict={features_placeholder: features,
#                                               labels_placeholder: labels})

# # Preprocess images
# expected_image_size = hub.get_expected_image_size(pretrained_model)
# def decode_and_resize_image(encoded, image_size=expected_image_size):
#     decoded = tf.image.decode_jpeg(encoded, channels=3)
#     decoded = tf.image.convert_image_dtype(decoded, tf.float32)
#     return tf.image.resize_images(decoded, image_size)
#
#
# # run images through pre-trained model
# encoded_images = tf.placeholder(tf.string, shape=[None])
# batch_images = tf.map_fn(decode_and_resize_image, encoded_images, dtype=tf.float32)
# features = pretrained_model(batch_images)


