import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from PIL import Image, ImageOps
from model import UNet


# PyTorch is channel first instead of channel last as in Keras
class LungSegmentationDataGen(Dataset):
    def __init__(self, dataset, root_dir, args, transforms=None):
        self.dataset = dataset
        self.root_dir = root_dir
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def preprocess_image(self, path):
        im = Image.open(path)
        im = ImageOps.grayscale(im)
        # parameterize height and width
        im = im.resize((self.args.height, self.args.width))
        img = np.array(im)
        img = img/255.
        img = np.expand_dims(img, axis=0)
        return img

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img_name, mask_name = sample.rstrip().split(",")

        image = self.preprocess_image(os.path.join(
            self.root_dir, "images", img_name.strip()))
        mask = self.preprocess_image(os.path.join(
            self.root_dir, "masks", mask_name.strip()))

        # Use albumentation package for augmenting images and mask with same transformations

        # numpy uses float64 as their default type, so call float() on these tensors before passing them to the TensorDataset
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


def dice_score(y_true, y_pred):
    smooth = 1
    y_true_flatten = y_true.view(-1)
    y_pred_flatten = y_pred.view(-1)
    intersection = 2 * torch.sum(y_true_flatten * y_pred_flatten)
    union = torch.sum(y_true_flatten) + torch.sum(y_pred_flatten)
    dice_score = (intersection + smooth)/(union + smooth)
    return dice_score


def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)


def train(args):
    dataset = open("dataset.csv", "r").readlines()
    train_set = dataset[:600]
    val_set = dataset[600:]
    root_dir = root_dir = "data/Lung_Segmentation/"

    train_data = LungSegmentationDataGen(train_set, root_dir, args)
    val_data = LungSegmentationDataGen(val_set, root_dir, args)

    train_dataloader = DataLoader(
        train_data, batch_size=5, shuffle=True, num_workers=4)

    val_dataloader = DataLoader(
        val_data, batch_size=5, shuffle=True, num_workers=4)

    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    dataset_sizes = {"train": len(train_set), "val": len(val_set)}

    print("dataset_sizes: {}".format(dataset_sizes))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())

    loss_train = []
    loss_valid = []

    current_mean_dsc = 0.0
    best_validation_dsc = 0.0

    epochs = args.epochs
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        dice_score_list = []
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                inputs, y_true = data
                inputs = inputs.to(device)
                y_true = y_true.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass with batch input
                    y_pred = model(inputs)

                    loss = dice_loss(y_true, y_pred)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # print("step: {}, train_loss: {}".format(i, loss))
                        loss_train.append(loss.item())

                        # calculate the gradients based on loss
                        loss.backward()

                        # update the weights
                        optimizer.step()

                    if phase == "val":
                        loss_valid.append(loss.item())
                        dsc = dice_score(y_true, y_pred)
                        print("step: {}, val_loss: {}, val dice_score: {}".format(i, loss, dsc))
                        dice_score_list.append(dsc.detach().numpy())

                if phase == "train" and (i + 1) % 10 == 0:
                    print("step:{}, train_loss: {}".format(i+1, np.mean(loss_train)))
                    loss_train = []
            if phase == "val":
                print("mean val_loss: {}".format(np.mean(loss_valid)))
                loss_valid = []
                current_mean_dsc = np.mean(dice_score_list)
                print("validation set dice_score: {}".format(current_mean_dsc))
                if current_mean_dsc > best_validation_dsc:
                    best_validation_dsc = current_mean_dsc
                    print("best dice_score on val set: {}".format(best_validation_dsc))
                    model_name = "unet_{0:.2f}.pt".format(best_validation_dsc)
                    torch.save(model.state_dict(), os.path.join(args.weights, model_name))


class Args(object):
    def __init__(self):
        self.batch_size = 2
        self.model_depth = 5
        self.width = 256
        self.root_filter_size = 32
        self.epochs = 100
        self.height = 256
        self.weights = "trained_model"


args = Args()
train(args)
