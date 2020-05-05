import numpy as np
import tensorflow as tf
import preprocessing
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling3D, AveragePooling2D
from skimage import io
import matplotlib.pyplot as plt
from tensorflow import keras

from matplotlib.image import imread
from skimage import color
from skimage.transform import resize
import h5py
import sys
import os

img_h = 200
img_w = 200

img_hd = int(img_h/4)
img_wd = int(img_w/4)

class R1Model(tf.keras.Model):
    def __init__(self):
        super(R1Model, self).__init__()

        self.checkpoint_path = "./r1_checkpoints/"

        self.optimizer = keras.optimizers.SGD()

        self.r1 = [
            # Block 1
            Conv2D(16, 9, 1, padding="same", name="block1_conv1", input_shape = (None, None, 1), kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)), #come back to input_shape if decide to not resize
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(32, 7, 1, padding="same", name="block2_conv1", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(16, 7, 1, padding="same", name="block3_conv1", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            Conv2D(8, 7, 1, padding="same", name="block3_conv2", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            Conv2D(1, 1, 1, padding="same", name="block3_conv3", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        ]

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.r1:
            img = layer(img)
        img = tf.clip_by_value(img, 0, 1)
        return img

    def loss_fn(self, labels, predictions):
        """ Loss function for the model. """
        sum_A = tf.math.reduce_sum(labels, axis=(1,2,3))
        sum_B = tf.math.reduce_sum(predictions, axis=(1,2,3))
        diff = tf.math.subtract(sum_A, sum_B)
        loss = tf.math.reduce_mean(tf.math.abs(diff))
        return loss
        

class R2Model(tf.keras.Model):
    def __init__(self):
        super(R2Model, self).__init__()

        self.checkpoint_path = "./r2_checkpoints/"

        self.optimizer = keras.optimizers.SGD()


        self.r2 = [
            # Block 1
            Conv2D(20, 7, 1, padding="same", name="block1_conv1", input_shape = (None, None, 1)), #come back to input_shape if decide to not resize
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(40, 5, 1, padding="same", name="block2_conv1"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(20, 5, 1, padding="same", name="block3_conv1"),
            Conv2D(10, 5, 1, padding="same", name="block3_conv2"),
            Conv2D(1, 1, 1, padding="same", name="block3_conv3")
        ]

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.r2:
            img = layer(img)
            img = tf.clip_by_value(img, 0, 1)
        return img

    def loss_fn(self, labels, predictions):
        """ Loss function for the model. """
        sum_A = tf.math.reduce_sum(labels, axis=(1,2,3))
        sum_B = tf.math.reduce_sum(predictions, axis=(1,2,3))
        diff = tf.math.subtract(sum_A, sum_B)
        loss = tf.math.reduce_mean(tf.math.abs(diff))
        return loss

class R3Model(tf.keras.Model):
    def __init__(self):
        super(R3Model, self).__init__()

        self.checkpoint_path = "./r3_checkpoints/"

        self.optimizer = keras.optimizers.SGD()


        self.r3 = [
            # Block 1
            Conv2D(24, 5, 1, padding="same", name="block1_conv1", input_shape = (None, None, 1)), #come back to input_shape if decide to not resize
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(48, 3, 1, padding="same", name="block2_conv1"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(24, 3, 1, padding="same", name="block3_conv1"),
            Conv2D(12, 3, 1, padding="same", name="block3_conv2"),
            Conv2D(1, 1, 1, padding="same", name="block3_conv3")
        ]

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.r3:
            img = layer(img)
            img = tf.clip_by_value(img, 0, 1)
        return img

    def loss_fn(self, labels, predictions):
        """ Loss function for the model. """
        sum_A = tf.math.reduce_sum(labels, axis=(1,2,3))
        sum_B = tf.math.reduce_sum(predictions, axis=(1,2,3))
        diff = tf.math.subtract(sum_A, sum_B)
        loss = tf.math.reduce_mean(tf.math.abs(diff))
        return loss

# class MaxModel(tf.keras.Model):
#     def __init__(self):
#         super(MaxModel, self).__init__()

#         self.max = [
#             AveragePooling2D(4, name="block1_pool", padding = "same", input_shape = (None, None, 1),  data_format='channels_last')
#         ]

#     def call(self, img):
#         """ Passes the image through the network. """

#         for layer in self.max:
#             img = layer(img)
#             img = tf.math.scalar_mul(16, img)

#         return img

def prepare_dataset(images, densities):
    # maxmodel = MaxModel()
    # maxmodel(tf.keras.Input(shape = (None, None, 1)))
    # maxmodel.summary()

    # maxmodel.compile(
    #     'sgd',
    #     loss=tf.keras.losses.MeanSquaredError()
    #     )
    
    # dens = []
    # for density in densities:
    #     dens.append(np.nan_to_num(density.reshape((density.shape[0], density.shape[1], 1))))
    
    # density_dataset = tf.data.Dataset.from_generator(lambda: dens, output_shapes=tf.TensorShape([None, None, 1]), output_types='float64')
    # density_dataset = density_dataset.batch(1)

    # new_densities = []
    # for density in dens: 
    #     new_densities.append(maxmodel.predict(x = density.reshape((1, density.shape[0], density.shape[1], 1))))
    # new_densities = maxmodel(density_dataset)

    # for i in range(10):
    #     print(np.sum(densities[i]), np.sum(new_densities[i]))

    data = []
    for i in range(len(images)):
        image = np.nan_to_num(images[i])/255
        density = np.nan_to_num(densities[i])
        im = image.reshape((image.shape[0], image.shape[1], 1))
        den = density.reshape((density.shape[0], density.shape[1], 1))
        data.append((im, den))

    return data

def main():

    #input image sets
    images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
    # images = images + (preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_B/train_data/images"))
    print("train inputs loaded")
    densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")
    # densities = densities + (preprocessing.density_patches("ShanghaiTech_PartB_Train/part_B/train_data/ground-truth-h5"))
    print("train maps loaded")
    images_test = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images")
    # images_test = images_test + (preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_B/test_data/images"))
    print("test inputs loaded")
    densities_test = preprocessing.density_patches("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5")
    # densities_test = densities_test + (preprocessing.density_patches("ShanghaiTech_PartB_Test/part_B/test_data/ground-truth-h5"))
    print("test maps loaded")

    train_data = prepare_dataset(images, densities) # returns tuples
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data, output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])), output_types=('float64', 'float64'))
    train_dataset = train_dataset.batch(1)
    print("train dataset loaded")

    test_data = prepare_dataset(images_test, densities_test) # returns tuples
    test_dataset = tf.data.Dataset.from_generator(lambda: test_data, output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])), output_types=('float64', 'float64'))
    test_dataset = test_dataset.batch(1)
    print("test dataset loaded")

    networks = [R1Model(), R2Model(), R3Model()]
    
    for model in networks:

    #model training begins here
        model(tf.keras.Input(shape = (None, None, 1)))
        checkpoint_path = model.checkpoint_path
        model.summary()

        model.compile(
            optimizer= tf.keras.optimizers.SGD(learning_rate = 0.0005, momentum = 0.9),
            loss=model.loss_fn
            )

        callback_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path + \
                        "weights.e{epoch:02d}",
                save_best_only=True,
                save_weights_only=True)
            ]

        model.fit(
            x = train_dataset,
            validation_data = test_dataset,
            epochs= hp.num_epochs,
            batch_size= None,
            callbacks= callback_list
        )

        print("done training")

        mae = 0.0
        i = 0
        for image, dens_map in test_dataset.as_numpy_iterator(): 
            pred = model.predict(x = image)
            mae += abs(np.sum(pred) - np.sum(dens_map))
            i += 1

        mae = mae/i
        # pred = model.predict(
        #     x =test_set,
        #     )

        # mae = 0
        # for i in range(len(pred)):
        #     mae += abs(np.sum(pred[i])-np.sum(density_test[i]))
        # mae = mae/len(pred)
        print(mae)

        print("done")

if __name__ == '__main__':
    main()