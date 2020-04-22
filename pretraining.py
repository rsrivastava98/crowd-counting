import numpy as np
import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling3D

class R1Model(tf.keras.Model):
    def __init__(self):
        super(R1Model, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.r1 = [
            # Block 1
            Conv2D(16, 9, 1, padding="same", name="block1_conv1"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(32, 7, 1, padding="same", name="block2_conv1"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(16, 7, 1, padding="same", name="block3_conv1"),
            Conv2D(8, 7, 1, padding="same", name="block3_conv2"),
            Conv2D(1, 1, 1, padding="same", name="block3_conv3"),
        ]

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.r1:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.MeanSquaredError(
            labels, predictions, from_logits=False)

class R2Model(tf.keras.Model):
    def __init__(self):
        super(R2Model, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.r2 = [
            # Block 1
            Conv2D(20, 7, 1, padding="same", name="block1_conv1"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(40, 5, 1, padding="same", name="block2_conv1"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(20, 5, 1, padding="same", name="block3_conv1"),
            Conv2D(10, 5, 1, padding="same", name="block3_conv2"),
            Conv2D(1, 1, 1, padding="same", name="block3_conv3"),
        ]

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.r2:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.MeanSquaredError(
            labels, predictions, from_logits=False)

class R3Model(tf.keras.Model):
    def __init__(self):
        super(R3Model, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.r3 = [
            # Block 1
            Conv2D(24, 5, 1, padding="same", name="block1_conv1"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(48, 3, 1, padding="same", name="block2_conv1"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(24, 3, 1, padding="same", name="block3_conv1"),
            Conv2D(12, 3, 1, padding="same", name="block3_conv2"),
            Conv2D(1, 1, 1, padding="same", name="block3_conv3"),
        ]

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.r3:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.MeanSquaredError(
            labels, predictions, from_logits=False)


class SwitchModel(tf.keras.Model):
    def __init__(self):
        super(SwitchModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.switch = [
            # Block 1
            Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        for layer in self.vgg16:
            layer.trainable = False
        # TODO: Write a classification head for our 15-scene classification task.
        #       Hint: The layers Flatten and Dense are essential here.
        self.head = []
        self.head.append(GlobalAveragePooling3D())
        self.head.append(Dense(512))
        self.head.append(Dense(3, activation='softmax'))

        # ============================================================================

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.switch:
            img = layer(img)

        for layer in self.head:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.categorical_crossentropy(
            labels, predictions, from_logits=False)
