"""
DS=~/Dropbox
INPUT=$DS/hand-segmented-fish-new-old-split
python3 unet.py train $INPUT --augment -l $DS/unet-logs -k $DS/segmentation-model.h5 -n firstrun

python3 unet.py train $INPUT --augment --show_training_data
"""
from __future__ import absolute_import, print_function, division
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, Callback)
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import os
from cv2 import resize
from skimage.io import imread, imsave
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.utils import set_trainable
from shutil import copyfile
from time import time
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tempfile import tempdir

tempdir = tempdir if tempdir else '/tmp'

try:
    from iotools import get_data_generators
    from visualize import visualize
except:
    from .iotools import get_data_generators
    from .visualize import visualize


class VisualValidation(Callback):
    # https://stackoverflow.com/questions/46587605

    def __init__(self, image_paths, output_dir):
        self.image_paths = image_paths
        self.outdir = output_dir
        self.out_imgdir = os.path.join(output_dir, 'images')
        self.out_segdir = os.path.join(output_dir, 'segmentations')
        if not os.path.exists(self.out_imgdir):
            os.makedirs(self.out_imgdir)
        if not os.path.exists(self.out_segdir):
            os.makedirs(self.out_segdir)

    def on_epoch_end(self, epoch, logs={}):
        for image in self.image_paths:
            name, ext = os.path.splitext(os.path.basename(image))
            tagged_name = name + '_%05d' % epoch
            img_out = os.path.join(self.out_imgdir, tagged_name + ext)
            seg_out = os.path.join(self.out_segdir, tagged_name + '.png')

            copyfile(image, img_out)
            predict(model=self.model, image_path=image, out_path=seg_out)


def get_callbacks(checkpoint_path=None, verbose=None, batch_size=None,
                  patience=None, logdir=None, run_name=None,
                  visual_validation_samples=None, steps_per_report=None):
    callbacks = dict()
    if checkpoint_path is not None:
        callbacks['ModelCheckpoint'] = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            verbose=int(verbose),
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)
    if patience is not None:
        callbacks['EarlyStopping'] = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=patience,
            verbose=int(verbose),
            mode='auto')
    if logdir is not None:
        callbacks['TensorBoard'] = TensorBoard(
            log_dir=os.path.join(logdir, run_name),
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=False,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            embeddings_data=None,
            update_freq=steps_per_report if steps_per_report else 'epoch')
    if visual_validation_samples is not None:
        callbacks['VisualValidation'] = VisualValidation(
            image_paths=visual_validation_samples,
            output_dir=os.path.join(logdir, run_name, 'visual_validation'))
    return callbacks


# def show_generated_pairs(generator, batch_size):
#     import cv2 as cv
#     for image, mask in generator:
#         for k in range(batch_size):
#             cmask = np.concatenate([mask[k]] * 3, axis=2)
#             rgb = np.flip(image[k], 2)
#             cv.imshow('', np.hstack((rgb, cmask)))
#             print(image[k].shape, mask[k].shape)
#             cv.waitKey(10)
#             input()
#             cv.destroyAllWindows()


def show_generated_pairs(generator, batch_size):
    import cv2 as cv
    for image_batch, mask_batch in generator:
        for image, mask in zip(image_batch, mask_batch):
            bgr = np.flip(image, axis=2)

            visual = visualize(bgr, mask)
            cv.imshow('', visual)
            print(image.shape, mask.shape)

            cv.waitKey(10)
            keypress = input()
            if keypress == 'q':
                cv.destroyAllWindows()
                return


def predict(model, image_path, out_path=None, backbone='resnet34',
            preprocessing_fcn=None, input_size=(256, 256)):
    if preprocessing_fcn is None:
        preprocessing_fcn = get_preprocessing(backbone)

    def quantize(mask):
        mask = mask * 255
        return mask.squeeze().astype('uint8')

    x = imread(image_path)
    x = resize(x, input_size)
    x = x / 255
    x = preprocessing_fcn(x)
    x = np.expand_dims(x, 0)
    y = model.predict(x)

    if out_path is not None:
        imsave(out_path, quantize(y))
    return y


def predict_all(model, data_dir, out_dir='results', backbone='resnet34',
                input_size=(256, 256), preprocessing_fcn=None):
    if preprocessing_fcn is None:
        preprocessing_fcn = get_preprocessing(backbone)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    images = [os.path.join(data_dir, 'val', 'images', fn) for fn in
              os.listdir(os.path.join(data_dir, 'val', 'images'))]

    for fn in images:
        name = os.path.splitext(os.path.basename(fn))[0]
        predict(model=model,
                image_path=fn,
                out_path=os.path.join(out_dir, name + '.png'),
                preprocessing_fcn=preprocessing_fcn,
                input_size=input_size)


def train(data_dir, model=None, backbone='resnet34', encoder_weights='imagenet',
          batch_size=2, all_layer_epochs=100, decode_only_epochs=2,
          logdir='logs', run_name='fish', verbose=2,
          patience=10, checkpoint_path='model_fish.h5', optimizer='adam',
          input_size=(256, 256), keras_augmentations=None,
          preprocessing_function_x=None, preprocessing_function_y=None,
          debug_training_data=False, debug_validation_data=False,
          preload=False, cached_preloading=False,
          visual_validation_samples=None, datumbox_mode=False,
          random_crops=False, learning_rate=None):
    # get data generators
    (training_generator, validation_generator,
     training_steps_per_epoch, validation_steps_per_epoch) = \
        get_data_generators(data_dir=data_dir,
                            backbone=backbone,
                            batch_size=batch_size,
                            input_size=input_size,
                            keras_augmentations=keras_augmentations,
                            preprocessing_function_x=preprocessing_function_x,
                            preprocessing_function_y=preprocessing_function_y,
                            preload=preload,
                            cached_preloading=cached_preloading,
                            random_crops=random_crops)

    # show images as they're input into model
    if debug_training_data:
        show_generated_pairs(training_generator, batch_size)
        return
    if debug_validation_data:
        show_generated_pairs(validation_generator, batch_size)
        return

    # initialize model
    if datumbox_mode and model is None:
        print('\n\nRunning in datumbox mode...\n\n')
        try:
            from datumbox_model import DatumboxUnet
        except:
            from .datumbox_model import DatumboxUnet
        model = DatumboxUnet(backbone_name=backbone,
                             encoder_weights=encoder_weights,
                             encoder_freeze=True)
    elif model is None:
        model = Unet(backbone_name=backbone,
                     encoder_weights=encoder_weights,
                     encoder_freeze=True)
    elif isinstance(model, str):
        model = load_model(model)

    if learning_rate is not None:
        if optimizer == 'adam':
            optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                             epsilon=None, decay=0.0, amsgrad=False)
        else:
            raise NotImplementedError(
                'Adjustable learning rate not implemented for %s.' % optimizer)
    model.compile(optimizer, loss=bce_jaccard_loss, metrics=[iou_score])
    # model.compile(optimizer, 'binary_crossentropy', ['binary_accuracy'])

    # get callbacks
    callbacks = get_callbacks(
        checkpoint_path=checkpoint_path,
        verbose=verbose,
        batch_size=batch_size,
        patience=patience,
        logdir=logdir,
        run_name=run_name,
        visual_validation_samples=visual_validation_samples,
        steps_per_report=training_steps_per_epoch)

    # train for `decoder_only_epochs` epochs with encoder frozen
    if decode_only_epochs:
        print('\n\nTraining decoder (only) for %s epochs...\n'
              '' % decode_only_epochs)
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            validation_steps=int(validation_steps_per_epoch),
                            steps_per_epoch=int(training_steps_per_epoch),
                            epochs=decode_only_epochs,
                            callbacks=list(callbacks.values()))

    # train all layers
    if all_layer_epochs:

        # refresh early stopping callback
        callbacks['EarlyStopping'] = \
            get_callbacks(patience=patience, verbose=verbose)['EarlyStopping']

        print('\n\nTraining all layers for %s epochs...\n' % all_layer_epochs)
        set_trainable(model)  # set all layers trainable and recompile model
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            validation_steps=int(validation_steps_per_epoch),
                            steps_per_epoch=int(training_steps_per_epoch),
                            epochs=all_layer_epochs,
                            callbacks=list(callbacks.values()),
                            initial_epoch=decode_only_epochs)
    return model


default_keras_augmentations = dict(rotation_range=0.2,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.05,
                                   zoom_range=0.05,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

if __name__ == '__main__':

    # # user defaults
    # DATA_DIR = '/home/andy/Desktop/hand_clipped_split'
    # RESULTS_DIR = '/mnt/datastore/unet-store/test-results-d'
    # CHECKPOINT_PATH = '/mnt/datastore/unet-store/model_fish_longrun.h5'
    # LOG_DIR = '/mnt/datastore/unet-store/logs'
    # EPOCHS = 100

    # os.system('mkdir %s/segmentations' % RESULTS_DIR)
    # os.system('mv %s/*.png %s/segmentations' % (RESULTS_DIR, RESULTS_DIR))
    # os.system('ln -s %s/val/images %s/images' % (DATA_DIR, RESULTS_DIR))

    # parse CLI arguments
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("command",
                      help="'train' or 'predict' (file or directory)")
    args.add_argument("input",
                      help="A file or directory.\nIf `command` is "
                           "'train', then this must be a directory "
                           "containing 'images' and 'segmentations' "
                           "subdirectories.\nIf `command` is 'predict', "
                           "then this can be a file or directory.")
    args.add_argument('-o', "--output",
                      help="Path to use for output file/directory.")
    args.add_argument("--epochs", default=100, type=int,
                      help="Number of training epochs.")
    args.add_argument("--decoder_only_epochs", default=2, type=int,
                      help="Number of epochs freeze the encoder for when "
                           "starting training.")
    args.add_argument("--batch_size", default=2, type=int,
                      help="Number of images per training batch.")
    args.add_argument('-l', "--logdir",
                      default=os.path.join(tempdir, 'unet-logs'),
                      help="Where to store logs.")
    args.add_argument('-k', "--checkpoint_path",
                      default=os.path.join(tempdir, 'unet-checkpoint.h5'),
                      help="Where to store logs.")
    args.add_argument('-n', "--run_name", default='unnamed_%s' % time(),
                      help="Name for this run/experiment.")
    args.add_argument("--verbosity", default=2, type=int,
                      help="Verbosity setting.")
    args.add_argument("--patience", default=10, type=int,
                      help="Patience for early stopping.")
    args.add_argument('-a', "--augment", default=False, action='store_true',
                      help="Invoke to use augmentation.")
    args.add_argument("--initial_model", default=None,
                      help="Keras model to start with.")
    args.add_argument("--backbone", default='resnet34',
                      help="Model (encoder) backbone.")
    args.add_argument("--input_size", nargs=2, default=(1024, 1024), type=int,
                      help="Input size (will resize if necessary).")
    args.add_argument("--show_training_data",
                      default=False, action='store_true',
                      help="Input size (will resize if necessary).")
    args.add_argument("--show_validation_data",
                      default=False, action='store_true',
                      help="Input size (will resize if necessary).")
    args.add_argument('-V', "--visual_validation_samples",
                      default=None, nargs='+',
                      help="Image to use for visual validation.  Output "
                           "stored in <logdir>/visual/<run_name>")
    args.add_argument("--datumbox", default=False, action='store_true',
                      help="Invoke when using datumbox_keras.")
    args.add_argument('-r', "--learning_rate", default=None, type=float,
                      help="Learning rate for optimizer.")
    args.add_argument(
        '-p', "--preload", default=False, action='store_true',
        help="If invoked, dataset will be preloaded.")
    args.add_argument(
        '-c', "--cached_preloading", default=False, action='store_true',
        help="Speed up preloading by looking for npy files stored in "
             "TMPDIR from previous runs.  This will speed things up "
             "significantly is only appropriate when you want to reuse "
             "the split dataset created in the last run.")
    args.add_argument(
        '-C', "--random_crops", default=0.0, type=float,
        help="Probability of sending random crop instead of whole image.")
    args = args.parse_args()

    if args.augment:
        keras_augmentations = default_keras_augmentations
    else:
        keras_augmentations = None

    if not args.run_name:
        args.run_name = 'unnamed_%s' % time()

    if args.preload or args.cached_preloading:
        print("\n\n Preloading requires unsplit dataset functionality to "
              "be implemented.\n\n")
        raise NotImplementedError

    # record user arguments in log directory
    run_logs_dir = os.path.join(args.logdir, args.run_name)
    if not os.path.exists(run_logs_dir):
        os.makedirs(run_logs_dir)
    with open(os.path.join(run_logs_dir, 'args.txt'), 'w+') as f:
        s = '\n'.join("%s: %s" % (key, val) for key, val in vars(args).items())
        f.write(s)

    if args.command == 'train':
        m = train(data_dir=args.input,
                  model=args.initial_model,
                  backbone=args.backbone,
                  batch_size=args.batch_size,
                  all_layer_epochs=args.epochs - args.decoder_only_epochs,
                  decode_only_epochs=args.decoder_only_epochs,
                  logdir=args.logdir,
                  run_name=args.run_name,
                  verbose=args.verbosity,
                  patience=args.patience,
                  checkpoint_path=args.checkpoint_path,
                  input_size=args.input_size,
                  encoder_weights='imagenet',
                  optimizer='adam',
                  keras_augmentations=keras_augmentations,
                  debug_training_data=args.show_training_data,
                  debug_validation_data=args.show_validation_data,
                  preload=args.preload,
                  cached_preloading=args.cached_preloading,
                  visual_validation_samples=args.visual_validation_samples,
                  datumbox_mode=args.datumbox,
                  random_crops=args.random_crops,
                  learning_rate=args.learning_rate)

    elif args.command == 'predict':
        m = load_model(args.checkpoint_path)
        if os.path.isdir(args.input):
            predict_all(model=m,
                        data_dir=args.input,
                        out_dir=args.output,
                        input_size=args.input_size,
                        backbone=args.backbone)
        else:
            predict(model=m,
                    image_path=args.input,
                    out_path=args.output,
                    backbone=args.backbone,
                    preprocessing_fcn=get_preprocessing(args.backbone),
                    input_size=args.input_size)
    else:
        raise ValueError('`command` = "%s" not understood.' % args.command)
