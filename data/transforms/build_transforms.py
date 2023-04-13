import torchvision.transforms as T
import albumentations as A
import cv2

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = A.Compose([
            A.HorizontalFlip(p=cfg.INPUT.PROB),
            A.VerticalFlip(p=cfg.INPUT.PROB),
            A.RandomRotate90(p=cfg.INPUT.PROB),
            A.IAAPerspective(p=cfg.INPUT.PROB),
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=35, p=cfg.INPUT.PROB, border_mode=cv2.BORDER_REFLECT),
            A.ChannelShuffle(p=0.4),
            A.RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=30, p=cfg.INPUT.PROB),
            A.MedianBlur(blur_limit=3, p=cfg.INPUT.PROB),
            A.ToGray(p=cfg.INPUT.PROB),
            A.OneOf([
                A.OpticalDistortion(p=cfg.INPUT.PROB),
                A.GridDistortion(p=cfg.INPUT.PROB),
                A.IAAPiecewiseAffine(p=cfg.INPUT.PROB),
                ]),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(p=cfg.INPUT.PROB),
                A.GaussNoise(p=cfg.INPUT.PROB)
                ]),
            A.OneOf([
                A.HueSaturationValue(25,25,20, p=cfg.INPUT.PROB),
                A.CLAHE(clip_limit=5, p=cfg.INPUT.PROB),
                A.RandomBrightnessContrast(),
                A.RandomGamma()
            ]),
            ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
