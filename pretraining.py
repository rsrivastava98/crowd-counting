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

        self.checkpoint_path = "./r1_checkpoints/"

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

        self.checkpoint_path = "./r2_checkpoints/"

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

        self.checkpoint_path = "./r3_checkpoints/"

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
    def __init__(self, ):
        super(SwitchModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.checkpoint_path = "./switch_checkpoints/"

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

#training function 
def train(model, datasets, checkpoint_path):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_categorical_accuracy:.4f}.h5",
            monitor='val_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(datasets)
    ]

    # Include confusion logger in callbacks if flag set
    # if ARGS.confusion:
    #     callback_list.append(ConfusionMatrixLogger(datasets))

    # Begin training
    model.fit(
        x=datasets.train_data, #update once we have data from preprocessing 
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )

def main(datasets):

    # datasets = None #assign datasets from preprocess
    networks = [R1Model(), R2Model(), R3Model(), SwitchModel()]

    #Model pretrain
    for model in networks:
        model(tf.keras.Input(shape=(img_size, img_size, 3)))
        checkpoint_path = model.checkpoint_path
        model.summary()

        # Compile model graph
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss_fn,
            metrics=["categorical_accuracy"])

        train(model, datasets, checkpoint_path)
    print("done")

if __name__ == '__main__':
    main()  

