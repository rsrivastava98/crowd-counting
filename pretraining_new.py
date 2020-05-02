import numpy as np
import tensorflow as tf
import preprocessing
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling3D
from skimage import io
import matplotlib.pyplot as plt

from matplotlib.image import imread
from skimage import color
from skimage.transform import resize

img_h = 200
img_w = 200

img_hd = int(img_h/4)
img_wd = int(img_w/4)

class R1Model(tf.keras.Model):
    def __init__(self):
        super(R1Model, self).__init__()

        self.checkpoint_path = "./r1_checkpoints/"

        self.r1 = [
            # Block 1
            Conv2D(16, 9, 1, padding="same", name="block1_conv1", input_shape = (img_h, img_w, 1)), #come back to input_shape if decide to not resize
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(32, 7, 1, padding="same", name="block2_conv1"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(16, 7, 1, padding="same", name="block3_conv1"),
            Conv2D(8, 7, 1, padding="same", name="block3_conv2"),
            Conv2D(1, 1, 1, padding="same", name="block3_conv3")
        ]

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.r1:
            img = layer(img)

        return img

def main():

    #input image sets
    images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
    images_test = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images")

    image_set =  np.zeros((len(images), img_h, img_w))
    for i, image in enumerate(images):
        im = resize(image, (img_h, img_w))
        image_set[i] = im

    image_set = image_set.reshape((len(image_set), img_h, img_w, 1))

    test_set =  np.zeros((len(images_test), img_h, img_w))
    for i, image in enumerate(images_test):
        im = resize(image, (img_h, img_w))
        test_set[i] = im

    test_set = test_set.reshape((len(test_set), img_h, img_w, 1))


    #ground truth sets
    densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")
    densities_test = preprocessing.density_patches("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5")

    density_set =  np.zeros((len(densities), img_hd, img_wd))
    for i, image in enumerate(densities):
        image = np.array(image)
        im = resize(image, (img_hd, img_wd))
        density_set[i] = im

    density_set = density_set.reshape((len(density_set), img_hd, img_wd, 1))

    density_test =  np.zeros((len(densities_test), img_hd, img_wd))
    for i, image in enumerate(densities_test):
        image = np.array(image)
        im = resize(image, (img_hd, img_wd))
        density_test[i] = im

    density_test = density_test.reshape((len(density_test), img_hd, img_wd, 1))
    
    #model training begins here
    model = R1Model()
    model(tf.keras.Input(shape = (img_h, img_w, 1)))
    checkpoint_path = model.checkpoint_path
    model.summary()

    model.compile(
        'sgd',
        loss=tf.keras.losses.MeanSquaredError()
        )

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}",
            save_best_only=True,
            save_weights_only=True)
        ]

    model.fit(
        x = image_set,
        y = density_set,
        validation_data = (test_set, density_test),
        epochs= hp.num_epochs,
        batch_size= None,
        callbacks= callback_list
    )

    print("done training")

    pred = model.predict(
        x =test_set,
        )

    mae = 0
    for i in range(len(pred)):
        mae += abs(np.sum(pred[i])-np.sum(density_test[i]))
    mae = mae/len(pred)
    print(mae)

    print("done")

if __name__ == '__main__':
    main()