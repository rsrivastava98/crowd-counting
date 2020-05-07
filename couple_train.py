# https://github.com/keras-team/keras/issues/4465
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import preprocessing
import sys
import new_pretraining as pt
import random
from vgg_model import VGGModel
import hyperparameters as hp
import new_pretraining 
import os

def train(model, train_data, test_data, checkpoint_path):
    """ Training routine. """
    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}.h5",
            save_best_only=True,
            save_weights_only=True)
        
        # ImageLabelingLogger(datasets)
    ]

    # Begin training
    model.fit(
        x=train_data,
        validation_data=test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )

def test(model, test_data):
    """ Testing routine. """
    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


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
    # train_dataset = train_dataset.shuffle(2700, reshuffle_each_iteration=True)
    print("train dataset loaded")

    test_data = prepare_dataset(images_test, densities_test) # returns tuples
    test_dataset = tf.data.Dataset.from_generator(lambda: test_data, output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])), output_types=('float64', 'float64'))
    test_dataset = test_dataset.batch(1)
    print("test dataset loaded")

    model = VGGModel()
    checkpoint_path = "./vgg_model_checkpoints/"

    # checkpoint_path = "./r1_checkpoints/checkpoint"
    model(tf.keras.Input(shape=(None, None, 1)))
    model.summary()

    model.load_weights("vgg16_weights.h5", by_name=True)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    

    train(model, train_dataset, test_dataset, checkpoint_path)

 

if __name__ == '__main__':
    main()

