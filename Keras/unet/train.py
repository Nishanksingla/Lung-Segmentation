import numpy as np
import os
import argparse
from PIL import Image, ImageOps
from keras.preprocessing.image import img_to_array
from model import unet
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

CHANNEL = 1


class LungSegmentationDataSet():
    # Data Generator
    # 1. Read Image and annotations.
    # 2. Preprocess image like resize and normalize
    # 3. Data Augmentation
    # 4. Create a batch
    # 5. Yield the batch

    def __init__(self, args, mode):
        self.args = args
        if mode == "train":
            self.root_dir = os.getenv("SM_CHANNEL_TRAIN")
        elif mode == "val":
            self.root_dir = os.getenv("SM_CHANNEL_EVAL")

        data_gen_args = dict(rotation_range=30,
                             zoom_range=[0.8, 1],
                             horizontal_flip=True)

        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args)

    def get_data(self):
        data = []
        annotations_path = os.path.join(self.root_dir, "annotations")
        annotations_files = os.listdir(annotations_path)
        print("list of annotations_files: {}".format(annotations_files))

        for annotation_filename in annotations_files:
            annotation_path = os.path.join(
                annotations_path, annotation_filename)
            f = open(annotation_path, 'r')
            annotations = f.readlines()
            data += annotations
            f.close()
        return data

    def preprocess_image(self, path):
        im = Image.open(path)
        im = ImageOps.grayscale(im)
        im = im.resize((self.args.height, self.args.width))  # parameterize height and width
        img = img_to_array(im)
        img = img/255.
        return img

    def generator(self, samples, apply_transform=False):
        input_batch = []
        output_batch = []
        batch_size = self.args.batch_size
        seed = 1
        while True:
            for line in samples:
                image_id, image_name, annotation_type, _, mask_name, mask_label, _, _ = line.rstrip().split(",")
                image_path = os.path.join(self.root_dir, "images", image_name)
                mask_path = os.path.join(self.root_dir, "masks", mask_name)
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    image = self.preprocess_image(image_path)
                    mask = self.preprocess_image(mask_path)
                    input_batch.append(image)
                    output_batch.append(mask)
                    if len(input_batch) == batch_size:
                        images = np.array(input_batch)
                        masks = np.array(output_batch)
                        if apply_transform:
                            image_gen = self.image_datagen.flow(images, shuffle=False, seed=seed, batch_size=batch_size)
                            mask_gen = self.mask_datagen.flow(masks, shuffle=False, seed=seed, batch_size=batch_size)
                            gen = zip(image_gen, mask_gen)
                            images, masks = next(gen)
                        input_batch = []
                        output_batch = []
                        yield (images, masks)
                else:
                    print("DOES NOT EXISTS image path: {}, mask path: {}".format(image_path, mask_path))


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_percentage', type=int, default=80)
    parser.add_argument('--root_filter_size', type=int, default=64)
    parser.add_argument('--model_depth', type=int, default=5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--model_dir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_decay', type=bool, default=False)
    parser.add_argument('--reduce_lr', type=bool, default=False)

    return parser.parse_known_args()


def exponential_decay(epoch, current_lrate=1e-4):
    print("Current LR: {}".format(current_lrate))
    lrate = pow(0.99, epoch) * current_lrate
    print("new LR: {}".format(lrate))
    return lrate


# Define Loss Function
    # you can use keras inbuilt loss functions for classifications like categorical_cross entropy
    # but might have to write a loss function for image segmentation like dice score calculation
def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)


# Define metrics
def dice_score(y_true, y_pred):
    eps = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = 2 * K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice_score = (intersection+eps) / (union + eps)
    return dice_score


def train(args):
    root_filter_size = args.root_filter_size
    model_depth = args.model_depth
    epochs = args.epochs

    train_obj = LungSegmentationDataSet(args, "train")
    val_obj = LungSegmentationDataSet(args, "val")

    train_data = train_obj.get_data()
    val_data = val_obj.get_data()

    train_generator = train_obj.generator(train_data)
    val_generator = val_obj.generator(val_data)

    train_steps_per_epoch = int(len(train_data) / args.batch_size)
    print("length of trian set: {}, train_steps_per_epoch: {}".format(len(train_data), train_steps_per_epoch))

    val_steps_per_epoch = int(len(val_data) / args.batch_size)
    print("length of val set: {}, val_steps_per_epoch: {}".format(len(val_data), val_steps_per_epoch))

    filepath = os.path.join(os.getenv("SM_MODEL_DIR", "./"), "best_model.h5")
    print("Model save path: {}".format(filepath))

    # Define any callbacks that you need like Learning Rate decrement, early stopping etc
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

    callbacks_list = [checkpoint]
    if args.lr_decay:
        lrate = LearningRateScheduler(exponential_decay)
        callbacks_list.append(lrate)
    if args.reduce_lr:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        callbacks_list.append(reduce_lr)

    # Define the model
    model = unet(root_filter_size, model_depth, n_class=1, input_size=(args.height, args.width, CHANNEL))

    # Compile the model with loss function, metrics and optimzer
    model.compile(optimizer='sgd', loss=dice_loss, metrics=[dice_score])

    print(model.summary())

    # fit the model with train and val data generator
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch
    )


if __name__ == "__main__":
    args, _ = parse_args()
    print(args)
    train(args)
