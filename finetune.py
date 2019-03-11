#!/usr/bin/env python
"""Finetune Inception V3 (pre-trained on ImageNet).

Important note
--------------
Currently, for much better results, use [the datumbox
keras fork](https://github.com/datumbox/keras@fork/keras2.2.4)
For details see
https://github.com/keras-team/keras/pull/9965
"""


from __future__ import print_function, division, absolute_import
import tensorflow as tf
from keras.metrics import sparse_categorical_accuracy
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.utils import plot_model
from sklearn.metrics import classification_report, balanced_accuracy_score
import numpy as np
import os
from time import time
from math import ceil
import tempfile


try:
    from utils import resize_images
    from iotools import image_preloader
    from metrics import pretty_confusion_matrix, top_2_error, top_3_error
    from augmentations import get_augmentation_fcn
except:
    from .utils import resize_images
    from .iotools import image_preloader
    from .metrics import pretty_confusion_matrix, top_2_error, top_3_error
    from .augmentations import get_augmentation_fcn

_tmp_storage_dir = os.path.join(tempfile.gettempdir(), 'finetune')
if not os.path.exists(_tmp_storage_dir):
    os.mkdir(_tmp_storage_dir)


# KERAS_AUGMENTATIONS = dict(shear_range=0.2,
#                      zoom_range=0.2,
#                      horizontal_flip=True)
KERAS_AUGMENTATIONS = dict()


def loss_fn(y_true, y_pred):
    # https://github.com/keras-team/keras/issues/7818
    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, tf.int32)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred))


# def acc_fn(y_true, y_pred):
#     # import ipdb; ipdb.set_trace()  ### DEBUG
#     # y_true = tf.squeeze(y_true)
#     # y_true = tf.cast(y_true, tf.int32)
#     return sparse_categorical_accuracy(y_true, y_pred)


class MetricsCallback(Callback):
    # https://github.com/keras-team/keras/issues/2548
    def __init__(self, data_generator, steps):
        self.datagen = data_generator
        self.steps = steps

    def on_epoch_end(self, batch, logs={}):
        loss, acc = self.model.evaluate_generator(self.datagen,
                                                  verbose=0,
                                                  steps=self.steps)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def create_model_using_tinycnn(n_classes, input_shape, input_tensor=None,
                               metrics=['accuracy', 'top_k_'],
                               train_with_logits=True):

    # setup metrics
    if 'top_k' in metrics:
        if n_classes < 10:
            metrics[metrics.index('top_k')] = top_2_error
        else:
            metrics[metrics.index('top_k')] = \
                'sparse_top_k_categorical_accuracy'

    # setup loss
    loss = loss_fn if train_with_logits else 'sparse_categorical_crossentropy'
    final_activation = None if train_with_logits else 'softmax'

    # setup model
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='valid',
                            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation=final_activation))
    model.compile(loss=loss, optimizer='adam', metrics=metrics)

    return model


def create_pretrained_model(n_classes, input_shape, input_tensor=None,
                            base='inceptionv3', base_weights='imagenet',
                            metrics=['accuracy', 'top_k'],
                            learning_rate=0.0001, momentum=0.9,
                            train_with_logits=False, n_freeze=0):

    # setup metrics
    if 'top_k' in metrics:
        if n_classes < 10:
            metrics[metrics.index('top_k')] = top_2_error
        else:
            metrics[metrics.index('top_k')] = \
                'sparse_top_k_categorical_accuracy'

    # setup loss
    loss = loss_fn if train_with_logits else 'sparse_categorical_crossentropy'
    final_activation = None if train_with_logits else 'softmax'

    # get the (headless) backbone
    if base == 'resnet50':
        base_model_getter = applications.resnet50.ResNet50
    elif base == 'vgg19':
        base_model_getter = applications.vgg19.VGG19
    elif base == 'inceptionv3':
        base_model_getter = applications.inception_v3.InceptionV3
    else:
        raise ValueError('`base = "%s"` not understood.' % base)
    base_model = base_model_getter(include_top=False,
                                   weights=base_weights,
                                   input_tensor=input_tensor,
                                   input_shape=input_shape,
                                   pooling='avg')

    inputs = base_model.input if input_tensor is None else input_tensor

    # put the top back on the model (pooling layer is already included)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(n_classes, activation=final_activation)(x)
    model = Model(inputs=inputs, outputs=outputs)

    # freeze some layers
    # n_freeze = 314 - 65  # fine-tune top two inception blocks + head
    if n_freeze:
        for layer in model.layers[:n_freeze]:
            layer.trainable = False
        for layer in model.layers[n_freeze:]:
            layer.trainable = True
    else:
        for layer in base_model.layers:
            layer.trainable = False

    # for k, layer in enumerate(model.layers):
    #     print(k, layer.name)
    # import ipdb; ipdb.set_trace()  ### DEBUG

    # compile
    model.compile(optimizer=SGD(lr=learning_rate, momentum=momentum),
                  loss=loss, metrics=metrics)
    # model.compile(optimizer='rmsprop', loss=loss, metrics=metrics)
    return model


def preloader(dataset_dir, img_shape, subset='', cached_preloading=False,
              storage_dir=_tmp_storage_dir, verbose=True):
    subset_storage = os.path.join(storage_dir, 'data_preloader_' + subset)
    if not os.path.exists(subset_storage):
        os.mkdir(subset_storage)
    image_dir = os.path.join(dataset_dir, subset) if subset else dataset_dir
    x, y, _, _, _, _, class_names = \
        image_preloader(image_directory=image_dir,
                        size=img_shape[:2],
                        image_depth=img_shape[2],
                        label_type='subdirectory',
                        pixel_labels_lookup=None,
                        exts=('.jpg', '.jpeg', '.png'),
                        normalize=False,
                        shuffle=True,  # keras doesn't shuffle before splitting
                        onehot=False,
                        testpart=0,
                        validpart=0,
                        whiten=False,
                        ignore_existing=not cached_preloading,
                        storage_directory=subset_storage,
                        save_split_sets=False,
                        verbose=verbose)
    return x, y, class_names


def get_training_generators(dataset_dir, img_shape, valpart, batch_size,
                            cifar10, augment, presplit, preload,
                            cached_preloading, verbose=2):

    # preload data
    if preload:
        if cifar10:
            from keras.datasets import cifar10
            (x, y), _ = cifar10.load_data()
            x = resize_images(x, img_shape)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        elif presplit and not cifar10:
            x_train, y_train, class_names = \
                preloader(dataset_dir, img_shape, 'train', cached_preloading,
                          verbose=verbose)
            x_val, y_val, _ = \
                preloader(dataset_dir, img_shape, 'val', cached_preloading,
                          verbose=verbose)
        else:
            x, y, class_names = \
                preloader(dataset_dir, img_shape, '', cached_preloading,
                          verbose=verbose)

    # setup augmentation
    keras_augmentations = dict()
    if augment:
        keras_augmentations.update(KERAS_AUGMENTATIONS)

    # create data generator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.0 if presplit else valpart,
        preprocessing_function=get_augmentation_fcn(augment, p=0.9),
        **keras_augmentations
    )
    if preload:
        if presplit:
            train_generator = train_datagen.flow(
                x=x_train, y=y_train, batch_size=batch_size)
            validation_generator = train_datagen.flow(
                x=x_val, y=y_val, batch_size=batch_size)
        else:
            train_generator = train_datagen.flow(
                x=x, y=y, batch_size=batch_size, subset='training')
            validation_generator = train_datagen.flow(
                x=x, y=y, batch_size=batch_size, subset='validation')
        n_train = len(train_generator.x)
        n_val = len(validation_generator.x)
    else:
        train_generator = train_datagen.flow_from_directory(
            directory=os.path.join(dataset_dir, 'train') if presplit else dataset_dir,
            target_size=img_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset=None if presplit else 'training')
        validation_generator = train_datagen.flow_from_directory(
            directory=os.path.join(dataset_dir, 'val') if presplit else dataset_dir,
            target_size=img_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            subset=None if presplit else 'validation')
        n_val = len(validation_generator.classes)
        n_train = len(train_generator.classes)
        class_names = [l for l in train_generator.class_indices]

    print('Training/Validation samples found: %s, %s' % (n_train, n_val))
    batches_per_epoch = int(ceil(n_train / batch_size))
    batches_per_val_epoch = int(ceil(n_val / batch_size))
    return (train_generator, validation_generator, class_names,
            batches_per_epoch, batches_per_val_epoch)


def get_testing_generator(dataset_dir, img_shape, batch_size, cifar10,
                          preload, cached_preloading, verbose=True):

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    if preload:
        if cifar10:
            from keras.datasets import cifar10
            _, (x, y) = cifar10.load_data()
            x = resize_images(x, img_shape)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            x, y, class_names = \
                preloader(dataset_dir, img_shape, 'test', cached_preloading,
                          verbose=verbose)
        test_generator = test_datagen.flow(x=x, y=y, batch_size=batch_size,
                                           shuffle=False)
        batches_per_test = int(ceil(len(x) / batch_size))
    else:
        test_generator = test_datagen.flow_from_directory(
            directory=os.path.join(dataset_dir, 'test'),
            target_size=img_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            interpolation="lanczos",
            shuffle=False)
        batches_per_test = int(ceil(len(test_generator.classes) / batch_size))
        class_names = [l for l in test_generator.class_indices]

    return test_generator, class_names, batches_per_test


def retrain(dataset_dir, base='inceptionv3', pretrained_weights='imagenet',
            img_shape=(299, 299, 3), valpart=0.2, batch_size=100, epochs=50,
            augment=True, checkpoint_path='checkpoint.h5', test_only=False,
            presplit=False, learning_rate=0.0001, momentum=0.9, n_freeze=312,
            logdir='logs', run_name=None, epochs_per_epoch=1, preload=False,
            cached_preloading=False, verbose=2, optimizer='sgd',
            patience=20):

    # parse arguments and fix or warn about conflicting arguments
    cifar10 = (dataset_dir == 'cifar10')
    if cifar10:
        preload = True
        presplit = False

    if checkpoint_path is None:
            d = dataset_dir.strip(os.sep).split(os.sep)[-1]
            checkpoint_path = "%s-%s.h5" % (args.base, d)
    elif checkpoint_path == 'off':
        checkpoint_path = None

    if presplit:
        valpart = 0.0
    if run_name is None:
        run_name = 'unnamed_' + str(time())

    if augment and cifar10:
        from warnings import warn
        warn("\n\nTo use augmentation, `dataset_dir` must be a directory of "
             "images.  To not get this warning, use to --no_augmentation "
             "flag.\n\n")

    (train_generator, validation_generator, class_names,
     batches_per_epoch, batches_per_val_epoch) = get_training_generators(
        dataset_dir, img_shape, valpart, batch_size, cifar10, augment,
        presplit, preload, cached_preloading, verbose=verbose)

    # create training callbacks for checkpointing, early stopping, and logging
    callbacks = []
    if checkpoint_path is not None:
        callbacks.append(ModelCheckpoint(checkpoint_path,
                                         monitor='val_loss',
                                         verbose=int(verbose),
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='auto',
                                         period=1))
    callbacks.append(EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=patience,
                                   verbose=int(verbose),
                                   mode='auto'))
    callbacks.append(TensorBoard(log_dir=os.path.join(logdir, run_name),
                                 histogram_freq=0,
                                 batch_size=batch_size,
                                 write_graph=False,
                                 write_grads=False,
                                 write_images=False,
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None,
                                 embeddings_data=None,
                                 update_freq='epoch'))
    # callbacks.append(MetricsCallback(validation_generator, batches_per_epoch))

    # compile model
    if base == 'tinycnn':
        model = create_model_using_tinycnn(n_classes=len(class_names),
                                           input_shape=img_shape,
                                           input_tensor=None)
    else:
        model = create_pretrained_model(
            n_classes=len(class_names),
            input_shape=img_shape,
            input_tensor=None,
            base=base,
            base_weights=None if (pretrained_weights == 'scratch') else 'imagenet',
            metrics=['accuracy'],
            learning_rate=learning_rate,
            momentum=momentum,
            n_freeze=n_freeze)

    if pretrained_weights not in ('imagenet', 'scratch'):
        model.load_weights(pretrained_weights)

    if verbose:
        # create nice graphviz graph visualization of model
        plot_model(model, to_file='model.png')

        # print the layer names
        for k, l in enumerate(model.layers):
            # print(k, list(l.__dict__.values())[12:])
            print(k, l.name, l.trainable)

    # train model
    if not test_only:
        # metrics = ['accuracy']
        # metrics = [acc_fn]
        metrics = ['accuracy', 'binary_accuracy', 'categorical_accuracy',
                   'sparse_categorical_accuracy']
        loss = 'sparse_categorical_crossentropy'
        checkpoint_epoch = 0
        stages = [n_freeze] if n_freeze else [311, 280, 249]
        for stage in stages:
            for layer in model.layers[:stage]:
                layer.trainable = False
            for layer in model.layers[stage:]:
                layer.trainable = True

            if optimizer.lower() == 'sgd':
                model.compile(optimizer=SGD(lr=learning_rate, momentum=momentum),
                              loss=loss, metrics=metrics)
            elif optimizer.lower() == 'rmsprop':
                model.compile(optimizer=RMSprop(lr=learning_rate),
                              loss=loss, metrics=metrics)
            elif optimizer.lower() == 'adam':
                model.compile(optimizer=Adam(lr=learning_rate),
                              loss=loss, metrics=metrics)
            elif optimizer.lower() == 'nadam':
                model.compile(optimizer=Nadam(),
                              loss=loss, metrics=metrics)
            else:
                raise Exception("`optimizer='%s'` not understood." 
                                "" % optimizer)

            history = model.fit_generator(
                train_generator,
                epochs=epochs,
                steps_per_epoch=batches_per_epoch * epochs_per_epoch,
                verbose=int(verbose),
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=batches_per_val_epoch,
                class_weight=None,
                max_queue_size=batch_size,
                workers=6,
                use_multiprocessing=True,
                shuffle=True,
                initial_epoch=checkpoint_epoch)
            checkpoint_epoch = np.argmax(history.history['val_acc'])
            model.load_weights(checkpoint_path)

    # score over test data (if saved, use checkpoint)
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path)

    if presplit or cifar10:
        test_generator, class_names, batches_per_test = \
            get_testing_generator(dataset_dir, img_shape, batch_size, cifar10,
                                  preload, cached_preloading, verbose=verbose)
    elif not test_only:
        test_generator = validation_generator
        batches_per_test = int(ceil(len(test_generator.y) / batch_size))
    else:
        raise ValueError(
            '`test_only` can only be true if `presplit` or `cifar10`.')

    print("\nTest Results:\n" + '~'*13)
    y_pred = model.predict_generator(generator=test_generator,
                                     steps=batches_per_test
                                     ).argmax(axis=1)
    metrics = model.evaluate_generator(generator=test_generator,
                                       steps=batches_per_test)
    for metric, val in zip(model.metrics_names, metrics):
        print(metric, val)

    # print confusion matrix and scikit-image classification report
    if not preload:
        y_test = test_generator.classes
    else:
        y_test = test_generator.y
    print('(unbalanced) Accuracy =', sum(y_test == y_pred)/len(y_test))
    print('(balanced) Accuracy =', balanced_accuracy_score(
        y_test, y_pred, sample_weight=None, adjusted=False))
    print('(adj. balanced) Accuracy =', balanced_accuracy_score(
        y_test, y_pred, sample_weight=None, adjusted=True))

    print('\nConfusion Matrix')
    print(pretty_confusion_matrix(y_test, y_pred, class_names, abbr=True))

    print('\nClassification Report')
    print(classification_report(y_test, y_pred, target_names=class_names))


if __name__ == '__main__':
    # parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        help="A split or unsplit dataset of images w/ subdirectory labels."
             "If --presplit flag is invoked,  `dataset_dir` must contain"
             "subdirectories, 'train' and 'val' (and optionally 'test')."
             "Use 'cifar10' to use the cifar10 dataset as a test.")
    parser.add_argument(
        '-s', "--presplit", default=False, action='store_true',
        help="The image directory is already split into train and test sets.")
    parser.add_argument(
        '-a', "--augment", default=False,
        help="Invoke to specify augmentation mode.")
    parser.add_argument(
        "--base", default='inceptionv3',
        help="Base model to use. Use 'tinycnn' to train a small CNN from "
             "scratch.")
    parser.add_argument(
        '--pretrained_weights', default='imagenet',
        help="Model weights (.h5) file to use start with. Omit this flag"
             "to use weights pretrained ImageNet or 'scratch' to "
             "train from scratch.")
    parser.add_argument(
        "--square", default=False, action='store_true',
        help="Use square images.")
    parser.add_argument(
        "--max_dim", default=299, type=int,
        help="Images will be resized to `max_dim` x relative_other_dim.  If "
             "`--square` flag is used, images will be resized to "
             "`max_dim` x `max_dim`.")
    parser.add_argument(
        "--channels", default=3, type=int,
        help="Number of channels to assume images will have (usually 1 or 3).")
    parser.add_argument(
        "--batch_size", default=32, type=int,
        help="Training and inference batch size.")
    parser.add_argument(
        "--epochs", default=200, type=int,
        help="Training epochs.")
    parser.add_argument(
        "--valpart", default=0.2, type=float,
        help="Fraction of data to use for validation.")
    parser.add_argument(
        "--checkpoint_path",
        default=os.path.join(_tmp_storage_dir, 'checkpoint.h5'),
        help="Where to save the model weights. Use 'off' to not save.")
    parser.add_argument(
        "--test_only", default=False, action='store_true',
        help="Where to save the model weights.  Defaults (roughly speaking) "
             "to '<base_model>-<dataset>.h5'.")
    parser.add_argument(
        "--optimizer", default='adam',
        help="Which optimizer to use.")
    parser.add_argument(
        "--learning_rate", default=0.0003, type=float,
        help="optimizer learning rate")
    parser.add_argument(
        "--momentum", default=0.0, type=float,
        help="SGD momentum term coefficient.  "
             "Only applies if `optimizer=='sgd'`")
    parser.add_argument(
        "--n_freeze", default=0, type=int,
        help="Number of (from bottom) layers to freeze.  Use 0 for head only.")
    parser.add_argument(
        "--logdir", default=os.path.join(_tmp_storage_dir, 'logs'),
        help="Where to store tensorboard logs.")
    parser.add_argument(
        "--run_name", default=None,
        help="Name to use for run in logs.")
    parser.add_argument(
        '-p', "--preload", default=False, action='store_true',
        help="If invoked, dataset will be preloaded.")
    parser.add_argument(
        '-c', "--cached_preloading", default=False, action='store_true',
        help="Speed up preloading by looking for npy files stored in "
             "TMPDIR from previous runs.  This will speed things up "
             "significantly is only appropriate when you want to reuse "
             "the split dataset created in the last run.")
    parser.add_argument(
        "--verbose", default=2, type=int,
        help="Tell me more about the model I'm running.")
    parser.add_argument(
        "--patience", default=20, type=int,
        help="Patience when early stopping.")
    args = parser.parse_args()

    if args.square:
        args.size = (args.size, args.size, args.channels)
    else:
        raise NotImplementedError
    retrain(dataset_dir=args.dataset_dir,
            base=args.base,
            pretrained_weights=args.pretrained_weights,
            img_shape=args.size,
            valpart=args.valpart,
            batch_size=args.batch_size,
            epochs=args.epochs,
            augment=args.augment,
            checkpoint_path=args.checkpoint_path,
            test_only=args.test_only,
            presplit=args.presplit,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            n_freeze=args.n_freeze,
            logdir=args.logdir,
            run_name=args.run_name,
            preload=args.preload,
            cached_preloading=args.cached_preloading,
            verbose=args.verbose,
            optimizer=args.optimizer,
            patience=args.patience)
