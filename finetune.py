#!/usr/bin/env python
"""Finetune Inception V3 (pre-trained on ImageNet)."""

import tensorflow as tf
from keras.metrics import sparse_top_k_categorical_accuracy
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.backend import resize_images
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import os
from time import time
from math import ceil

try:
    from load_dataset import load_split_data
    from andnn_util import Timer
    from andnn.iotools import image_preloader
except:
    from .load_dataset import load_split_data
    from .andnn_util import Timer
    from .andnn.iotools import image_preloader


def top_3_error(y_true, y_pred):
    return 1 - sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_error(y_true, y_pred):
    return 1 - sparse_top_k_categorical_accuracy(y_true, y_pred, k=2)


def loss_fn(y_true, y_pred):
    # https://github.com/keras-team/keras/issues/7818
    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, tf.int32)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred))


def accuracy_fn(y_true, y_pred):
    # https://github.com/keras-team/keras/issues/7818
    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.argmax(y_pred, 1)
    correct_predictions = tf.equal(y_pred, y_true)
    return tf.reduce_mean(tf.cast(correct_predictions, "float"))


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
            metrics[metrics.index('top_k')] = 'sparse_top_k_categorical_accuracy'

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
            metrics[metrics.index('top_k')] = 'sparse_top_k_categorical_accuracy'

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
    # n_freeze = 314 - 65  # train top two inception blocks + head
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


def main(dataset_dir, base='inceptionv3', pretrained_weights=None,
         img_shape=(299, 299, 3), testpart=0.0, valpart=0.2,
         batch_size=100, epochs=50, samples_per_class=10**3, npy_data=False,
         cifar10=False, top_only_stage=False, augment=True,
         checkpoint_path='checkpoint.h5', test_only=False, presplit=False,
         learning_rate=0.0001, momentum=0.9, n_freeze=312, logdir='logs',
         run_name=None, epochs_per_epoch=1, preload=False):

    if run_name is None:
        run_name = 'unnamed_' + str(time())

    # warn if arguments conflict
    if augment and (npy_data or cifar10):
        from warnings import warn
        warn("\n\nTo use augmentation, `dataset_dir` must be a directory of "
             "images.  To not get this warning, use to --no_augmentation "
             "flag.\n\n")

    # load data
    if augment:
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           validation_split=valpart)
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           validation_split=valpart)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    if preload:
        if cifar10:
            from keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x = np.vstack((x_train, x_test))
            y = np.vstack((y_train, y_test))
            x = resize_images(x)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            x, y, _, _, _, _, class_names = \
                image_preloader(dataset_dir,
                                size=img_shape[:2],
                                image_depth=img_shape[2],
                                label_type='subdirectory',
                                pixel_labels_lookup=None,
                                exts=('.jpg', '.jpeg', '.png'),
                                normalize=False,
                                shuffle=False,
                                onehot=False,
                                testpart=0,
                                validpart=0,
                                whiten=False,
                                ignore_existing=False,
                                storage_directory='/tmp/data-preloader/',
                                save_split_sets=False)
        train_generator = train_datagen.flow(
            x=x, y=y, batch_size=batch_size, subset='training')
        validation_generator = train_datagen.flow(
            x=x, y=y, batch_size=batch_size, subset='validation')
        batches_per_epoch = int(ceil(len(train_generator.x) / batch_size))
        batches_per_val_epoch = \
            int(ceil(len(validation_generator.x) / batch_size))
    else:
        train_generator = train_datagen.flow_from_directory(
            os.path.join(dataset_dir, 'train') if presplit else dataset_dir,
            target_size=img_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            interpolation="lanczos",
            subset=None if presplit else 'training')
        validation_generator = train_datagen.flow_from_directory(
            os.path.join(dataset_dir, 'val') if presplit else dataset_dir,
            target_size=img_shape[:2],
            batch_size=batch_size,
            class_mode='sparse',
            interpolation="lanczos",
            subset=None if presplit else 'validation')
        batches_per_epoch = \
            int(ceil(len(train_generator.classes) / batch_size))
        batches_per_val_epoch = \
            int(ceil(len(validation_generator.classes) / batch_size))
        class_names = [l for l in train_generator.class_indices]

    # create training callbacks for checkpointing, early stopping, and logging
    callbacks = []
    if checkpoint_path is not None:
        callbacks.append(ModelCheckpoint(checkpoint_path,
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='auto',
                                         period=1))
    callbacks.append(EarlyStopping(monitor='val_acc',
                                   min_delta=0,
                                   patience=20,
                                   verbose=1,
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
        base_weights = None if (pretrained_weights == 'scratch') else 'imagenet'
        model = create_pretrained_model(n_classes=len(class_names),
                                        input_shape=img_shape,
                                        input_tensor=None,
                                        base=base,
                                        base_weights=base_weights,
                                        metrics=['accuracy'],
                                        learning_rate=learning_rate,
                                        momentum=momentum,
                                        n_freeze=n_freeze)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    plot_model(model, to_file='model.png')

    for k, l in enumerate(model.layers):
        # print(k, list(l.__dict__.values())[12:])
        print(k, l.name, l.trainable)
    # import ipdb; ipdb.set_trace()  ### DEBUG

    # train model
    if not test_only:
        # stage 1 fine-tuning (top only)
        if top_only_stage:
            top_only_epochs = 3
            print("\nStage 1 training (top only)...\n")
            stage1_history = model.fit_generator(
                train_generator,
                epochs=top_only_epochs,
                steps_per_epoch=batches_per_epoch * epochs_per_epoch,
                verbose=1,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=batches_per_val_epoch,
                class_weight=None,
                max_queue_size=10,
                workers=2,
                use_multiprocessing=True,
                shuffle=True,
                initial_epoch=0)
        else:
            top_only_epochs = 0

            if top_only_stage:
                print("\nStage 2 training (last two inception "
                      "blocks + top)...\n")
        history = model.fit_generator(
            train_generator,
            epochs=epochs - top_only_epochs,
            steps_per_epoch=batches_per_epoch * epochs_per_epoch,
            verbose=1,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=batches_per_val_epoch,
            class_weight=None,
            max_queue_size=10,
            workers=2,
            use_multiprocessing=True,
            shuffle=True,
            initial_epoch=top_only_epochs)

    # score over test data
    print("Test Results:\n" + '='*13)
    y_pred = model.predict_generator(generator=validation_generator,
                                     steps=batches_per_val_epoch
                                     ).argmax(axis=1)
    metrics = model.evaluate_generator(generator=validation_generator,
                                       steps=batches_per_val_epoch)
    for metric, val in zip(model.metrics_names, metrics):
        print(metric, val)

    # print confusion matrix and scikit-image classification report
    if not preload:
        y_test = validation_generator.classes
    else:
        y_test = validation_generator.y
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=class_names)
    cm.index = class_names
    print('Confusion Matrix')
    print(cm)
    print('Classification Report')
    print(classification_report(y_test, y_pred, target_names=class_names))

    # # Plot training & validation accuracy values
    # import matplotlib; matplotlib.use('agg')  # for when running over SSH
    # import matplotlib.pyplot as plt

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Training', 'Validation'], loc='upper left')
    # plt.savefig('history-accuracy_' + run_name + '.png')
    #
    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Training', 'Validation'], loc='upper left')
    # plt.savefig('history-validation_' + run_name + '.png')


if __name__ == '__main__':
    # parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        help="If --npy invoked, an (unsplit) directory of NPY files.  "
             "Otherwise a split dataset of image files containing two "
             "subdirectories, 'train' and 'test' each containing a "
             "subdirectory for each class.  Use 'cifar10' to use the "
             "cifar10 dataset.")
    parser.add_argument(
        "--presplit", default=False, action='store_true',
        help="The image directory is already split into train and test sets.")
    parser.add_argument(
        "--no_augmentation", default=False, action='store_true',
        help="Invoke to prevent augmentation.")
    parser.add_argument(
        "--base", default='inceptionv3',
        help="Base model to use. Use 'tinycnn' to train a small CNN from "
             "scratch.")
    parser.add_argument(
        "--npy", default=False, action='store_true',
        help="`dataset_dir` is directory of NPY files.")
    parser.add_argument(
        '--pretrained_weights', default=None,
        help="Model weights (.h5) file to use start with. Omit this flag"
             "to use weights pretrained ImageNet or 'scratch' to "
             "train from scratch.")
    parser.add_argument(
        "--size", default=299, type=int,
        help="Images will be resized to `size` x `size`.")
    parser.add_argument(
        "--channels", default=3, type=int,
        help="Number of channels to assume images will have (usually 1 or 3).")
    parser.add_argument(
        "--batch_size", default=100, type=int,
        help="Training and inference batch size.")
    parser.add_argument(
        "--epochs", default=50, type=int,
        help="Training epochs.")
    parser.add_argument(
        "--samples_per_class", default=10**3, type=int,
        help="Number of samples to include per class (ignored unless "
             "--npy flag invoked).")
    parser.add_argument(
        "--testpart", default=0.0, type=float,
        help="Fraction of data to use for test set.")
    parser.add_argument(
        "--valpart", default=0.2, type=float,
        help="Fraction of data to use for validation.")
    parser.add_argument(
        "--top_only_stage", default=False, action='store_true',
        help="Train for a few epochs on only the top of the model.")
    parser.add_argument(
        "--checkpoint_path", default='off',
        help="Where to save the model weights. Use 'off' to not save.")
    parser.add_argument(
        "--test_only", default=False, action='store_true',
        help="Where to save the model weights.  Defaults (roughly speaking) "
             "to '<base_model>-<dataset>.h5'.")
    parser.add_argument(
        "--learning_rate", default=0.0001, type=float,
        help="SGD learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float,
        help="SGD momentum term coefficient")
    parser.add_argument(
        "--n_freeze", default=0, type=int,
        help="Number of (from bottom) layers to freeze.  Use 0 for head only.")
    parser.add_argument(
        "--logdir", default='logs',
        help="Where to store tensorboard logs.")
    parser.add_argument(
        "--run_name", default=None,
        help="Name to use for run in logs.")
    parser.add_argument(
        "--preload", default=False, action='store_true',
        help="If invoked, dataset will be preloaded.")
    args = parser.parse_args()

    _image_shape = (args.size, args.size, args.channels)

    if args.dataset_dir == 'cifar10':
        args.dataset_dir = None
        _cifar10 = True
        args.preload = True
    else:
        _cifar10 = False

    if args.checkpoint_path is None:
        if _cifar10:
            args.checkpoint_path = "%s-cifar10.h5" % args.base
        else:
            d = args.dataset_dir.strip(os.sep).split(os.sep)[-1]
            args.checkpoint_path = \
                "%s-%s.h5" % (args.base, d)
    elif args.checkpoint_path == 'off':
        args.checkpoint_path = None

    from pprint import pprint
    pprint(vars(args))

    main(dataset_dir=args.dataset_dir,
         base=args.base,
         pretrained_weights=args.pretrained_weights,
         img_shape=_image_shape,
         testpart=args.testpart,
         valpart=args.valpart,
         batch_size=args.batch_size,
         epochs=args.epochs,
         samples_per_class=args.samples_per_class,
         npy_data=args.npy,
         cifar10=_cifar10,
         top_only_stage=args.top_only_stage,
         augment=not args.no_augmentation,
         checkpoint_path=args.checkpoint_path,
         test_only=args.test_only,
         presplit=args.presplit,
         learning_rate=args.learning_rate,
         momentum=args.momentum,
         n_freeze=args.n_freeze,
         logdir=args.logdir,
         run_name=args.run_name,
         preload=args.preload)
