
import os
import cv2
import csv
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import matplotlib.pyplot as plt

IM_H = 128
IM_W = 128

class Dataset(BaseDataset):

    CLASS_FILE = "_classes.csv"

    def __init__(self, images_dir,
                 classes=None, 
                 augmentation=None, 
                 preprocessing=None):
        
        self.images_dir = images_dir
        
        self.ids = os.listdir(images_dir)
        self.img_ids = [s for s in self.ids if ("_mask" not in s and "_classes" not in s)]
        self.mask_ids = [s for s in self.ids if ("_mask" in s and "_classes" not in s)]
        self.images_fps = [os.path.join(images_dir, image_id).replace("\\", "/") for image_id in self.img_ids]
        self.masks_fps = [os.path.join(images_dir, image_id).replace("\\", "/") for image_id in self.mask_ids]
        
        # convert str names to class values on masks
        self.class_values, self.class_names = self.get_classes()

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def get_classes(self):
        pixel_values = []
        classes = []

        file_path = os.path.join(self.images_dir, self.CLASS_FILE).replace("\\", "/")

        with open(file_path, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                pixel_value = int(row['Pixel Value'])
                class_name = row[' Class']
                pixel_values.append(pixel_value)
                classes.append(class_name)

        return pixel_values, classes

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i]) # [H, W, 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0) # [H, W]
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Image [C, H, W] -> 0.-255. float32 -> RGB format
        # Mask [Classes, H, W] -> 0. - 1. float32 -> RGB format
        # cv2.imshow("image", image.transpose(1, 2, 0))
        # cv2.imshow("mask", mask.transpose(1, 2, 0))
        # cv2.waitKey(0)
        image = image/255.
        return image, mask
    
    def __len__(self):
        return len(self.img_ids)
    
def to_tensor(x, **kwargs):
    x = x.transpose(2, 0, 1).astype('float32')
    # x = torch.as_tensor(x, device=torch.device('cuda'))
    return x
    

def get_training_augmentation():
    train_transform = [

        albu.Resize(IM_H, IM_W),

        albu.HorizontalFlip(p=0.5),

        albu.RandomGamma(p=0.7),

        albu.Blur(blur_limit=3, p=0.7),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(IM_H, IM_W),
    ]
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)