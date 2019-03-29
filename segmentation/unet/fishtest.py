# from model import *
# from data import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import skimage.transform as trans
from segmentation_models.backbones import get_preprocessing


_model_checkpoint = 'model_fish.h5'
data_dir = '/home/andy/Desktop/hand_clipped_split'


def adjustData(img,mask,flag_multi_class,num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    else:
        if (np.max(img) > 1):
            img = img / 255
        if (np.max(mask) > 1):
            mask = mask /255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
    return (img,mask)
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1, preprocessing_function_x=None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict, preprocessing_function=preprocessing_function_x)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

# myGene = trainGenerator(2,data_dir + '/train',
#                         'images','segmentations',
#                         data_gen_args,save_to_dir = None)
#
# val_gen = trainGenerator(2,data_dir + '/val',
#                         'images','segmentations',
#                          dict(),save_to_dir = None)
#
model_checkpoint = ModelCheckpoint(_model_checkpoint,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True)

def get_datagen(backbone='resnet34'):


    preprocessing_fn = get_preprocessing(backbone)
    # x = preprocessing_fn(x)

    myGene = trainGenerator(2,data_dir + '/train',
                        'images','segmentations',
                        data_gen_args,save_to_dir = None,
                        preprocessing_function_x=preprocessing_fn)

    val_gen = trainGenerator(2,data_dir + '/val',
                        'images','segmentations',
                         dict(),save_to_dir = None,
                         preprocessing_function_x=preprocessing_fn)
    return myGene, val_gen

# def train():
#     #os.environ["CUDA_VISIBLE_DEVICES"] = "0"



#     model = unet()
#     model_checkpoint = ModelCheckpoint(_model_checkpoint,
#                                        monitor='val_loss',
#                                        verbose=1,
#                                        save_best_only=True)
#     model.fit_generator(myGene,
#                         validation_data=val_gen,
#                         validation_steps=1,
#                         steps_per_epoch=300,
#                         epochs=1000,
#                         callbacks=[model_checkpoint])

#     # testGene = testGenerator(data_dir + "/val")
#     # results = model.predict_generator(testGene,30,verbose=1)
#     # saveResult(_storage,results)

#     model.save(_model_checkpoint)
#     return model


def test(model, out_dir='results', backbone='resnet34'):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    import skimage.io as io

    def loadimg(image_path, target_size = (256, 256), flag_multi_class=False,
                as_gray=False):
        img = io.imread(image_path, as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        # if not flag_multi_class:
        #     img = np.expand_dims(img, -1)
        # img = np.expand_dims(img, 0)
        return img

    imgs = [os.path.join(data_dir, 'val', 'images', fn) for fn in
            os.listdir(os.path.join(data_dir, 'val', 'images'))]

    preprocessing_fn = get_preprocessing(backbone)
    for fn in imgs:
        x = np.expand_dims(loadimg(fn), 0)
        x = preprocessing_fn(x)
        y = model.predict(x)
        s = y - y.min()
        s = s/s.max() * 255

        name = os.path.splitext(os.path.basename(fn))[0]
        io.imsave(os.path.join(out_dir, name + '.png'), s.squeeze().astype('uint8'))


def train2(sm, backbone='resnet34'):
    
    from segmentation_models.utils import set_trainable

    myGene, val_gen = get_datagen(backbone)

    model = sm(backbone_name=backbone, encoder_weights='imagenet', freeze_encoder=True)
    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

    # pretrain model decoder
    # model.fit(x, y, epochs=2)
    model.fit_generator(myGene,
                        validation_data=val_gen,
                        validation_steps=1,
                        steps_per_epoch=300,
                        epochs=2,
                        callbacks=[model_checkpoint])

    # release all layers for training
    set_trainable(model) # set all layers trainable and recompile model

    # continue training
    # model.fit(x, y, epochs=100)
    model.fit_generator(myGene,
                        validation_data=val_gen,
                        validation_steps=1,
                        steps_per_epoch=300,
                        # epochs=100,
                        epochs=2,
                        callbacks=[model_checkpoint])
    return model


def train3(sm, backbone='resnet34'):



    myGene, val_gen = get_datagen(backbone)

    # prepare model
    model = sm(backbone_name=backbone, encoder_weights='imagenet')
    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

    # train model
    model.fit_generator(myGene,
                        validation_data=val_gen,
                        validation_steps=1,
                        steps_per_epoch=300,
                        # epochs=100,
                        epochs=1,
                        callbacks=[model_checkpoint])
    return model


if __name__ == '__main__':

    from segmentation_models import FPN, Unet
    # m = train2(FPN)
    # test(m, 'results-fpn')

    m = train2(Unet)
    test(m, 'results-unet')

    m = train3(Unet)
    test(m, 'results3-2')

    # import ipdb; ipdb.set_trace()  ### DEBUG
