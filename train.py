import os
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_handler import Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.encoders import get_preprocessing_fn

device = 'cuda'

# initialize loss and metrics
loss = utils.losses.DiceLoss()
metrics = [
    utils.metrics.IoU(threshold=0.5),
]
HOME_DIR = os.getcwd()
DATA_DIR = os.path.join(HOME_DIR, "data").replace("\\", "/")
CLASSES = ['background', 'ball', 'goal']

train_imgs = os.path.join(DATA_DIR, "train").replace("\\", "/")
val_imgs = os.path.join(DATA_DIR, "valid").replace("\\", "/")
test_imgs = os.path.join(DATA_DIR, "test").replace("\\", "/")

# Initialize dataset and dataloaders
def train():
    # Initialize UNet model
    model = smp.Unet(
        encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=len(CLASSES),                      # model output channels (number of classes in your dataset)
        activation="sigmoid"
    )
    preprocess = get_preprocessing_fn('mobilenet_v2', pretrained='imagenet')

    train_data = Dataset(
        train_imgs, 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess),
        classes=CLASSES,
    )

    val_data = Dataset(
        val_imgs, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001)
    ])

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
        optimizer=optimizer
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = 0
    epochs = 40
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, 'models/best_model.pth')
            print('Model saved!')

        if i == 25:
            # adjust lr
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

def evaluate(model_path):
    model = torch.load(model_path)
    preprocess = get_preprocessing_fn('mobilenet_v2', pretrained='imagenet')
    test_epoch = utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=device,
    )
    test_dataset = Dataset(
        test_imgs,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocess),
        classes=CLASSES,
    )
    test_dataloader = DataLoader(test_dataset)
    logs = test_epoch.run(test_dataloader)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if image.shape[0] ==3:
            image = np.reshape(image, (image.shape[2], image.shape[1], image.shape[0]))
        plt.imshow(image)
    plt.show()

def visual_eval(model_path):
    model = torch.load(model_path)
    preprocess = get_preprocessing_fn('mobilenet_v2', pretrained='imagenet')
    test_dataset_vis = Dataset(
        test_imgs,
        classes=CLASSES,
    )

    test_dataset = Dataset(
        test_imgs,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocess),
        classes=CLASSES,
    )

    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        gt_mask_vis = test_dataset_vis[n][1].astype('uint8')
        image, gt_mask = test_dataset[n]


        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask_vis, 
            predicted_mask=pr_mask
        )

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", default=False, action="store_true")
    parser.add_argument("-m", "--model_path", dest="model_path", default=None)
    parser.add_argument("-e", "--eval", dest="eval", default=False, action="store_true")
    parser.add_argument("-v", "--visualize", dest="visualize", default=False, action="store_true")
    opts = parser.parse_args()
    if opts.train:
        train()
    elif opts.eval:
        evaluate(opts.model_path)
    elif opts.visualize:
        visual_eval(opts.model_path)

# dataset = Dataset(test_imgs, classes=['car'])

# image, mask = dataset[4] # get some sample
# print(mask.shape)
# print(mask.squeeze().shape)
# visualize(
#     image=image, 
#     cars_mask=mask.squeeze(),
# )