import albumentations as al


def get_augmentation_fcn(mode, p=0.75):
    if not mode or mode.lower() == 'none':
        return None
    augmentation_dict = {
        'no_interpolation_necessary':
            al.OneOf([al.RandomRotate90(p=1.),
                      al.Flip(p=1.)]),
        'interpolation_necessary':
            al.OneOf([al.Rotate(p=1.),
                      al.RandomScale(p=1.),
                      al.ShiftScaleRotate(p=1.)]),
        'affine': al.Compose([al.ShiftScaleRotate(p=1.),
                              al.HorizontalFlip(p=0.5)]),
        'rot': al.Rotate(p=1.),
        'rot90': al.RandomRotate90(p=1.),
        'flip': al.Flip(p=1.),
        'hflip': al.HorizontalFlip(p=1.),
        'vflip': al.VerticalFlip(p=1.),
        'scale': al.RandomScale(p=1.),
        'ssr': al.ShiftScaleRotate(p=1.),
        'strong': al.Compose([
                    # al.RandomRotate90(),
                    # al.Flip(),
                    # al.Transpose(),
                    al.OneOf([
                        al.IAAAdditiveGaussianNoise(),
                        al.GaussNoise(),
                    ], p=0.2),
                    al.OneOf([
                        al.MotionBlur(p=0.2),
                        al.MedianBlur(blur_limit=3, p=0.1),
                        al.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                    al.ShiftScaleRotate(shift_limit=0.0625,
                                        scale_limit=0.2,
                                        rotate_limit=10,
                                        p=0.2),
                    # al.OneOf([
                    #     al.OpticalDistortion(p=0.3),
                    #     al.GridDistortion(p=0.1),
                    #     al.IAAPiecewiseAffine(p=0.3),
                    # ], p=0.2),
                    al.OneOf([
                        # al.CLAHE(clip_limit=2),
                        al.IAASharpen(),
                        al.IAAEmboss(),
                        al.RandomBrightnessContrast(),
                    ], p=0.3),
                    al.HueSaturationValue(p=0.3),
                ], p=p)
    }

    def aug_fcn(x):
        try:
            return augmentation_dict[mode](image=x)['image']
        except Exception as e:
            print("Exception caught in augmentation stage:", e)
            return x

    return aug_fcn
