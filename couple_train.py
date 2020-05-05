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

def train(model, train_data, test_data, checkpoint_path):
    """ Training routine. """
    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-",
            save_best_only=True,
            save_weights_only=True)
        
        # ImageLabelingLogger(datasets)
    ]

    # Include confusion logger in callbacks if flag set
    # if ARGS.confusion:
    #     callback_list.append(ConfusionMatrixLogger(datasets))

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
    # datasets = Datasets(ARGS.data, ARGS.task)

    model = VGGModel()
    checkpoint_path = "./vgg_model_checkpoints/"
    model(tf.keras.Input(shape=(224, 224, 3)))
    model.summary()

    # Don't load pretrained vgg if loading checkpoint
    # if ARGS.load_checkpoint is None:
    #     # model.load_weights(ARGS.load_vgg, by_name=True)
    #     model.load_weights('/vgg16_imagenet.h5', by_name=True)


    # if ARGS.load_checkpoint is not None:
    #     model.load_weights(ARGS.load_checkpoint)

    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)

    # Compile model graph
    #input image sets
    # images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
    # print("train inputs loaded")
    # densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")
    # print("train maps loaded")
    # images_test = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images")
    # print("test inputs loaded")
    # densities_test = preprocessing.density_patches("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5")
    # print("test maps loaded")

    # train_data = pt.prepare_dataset(images, densities) # returns tuples
    # train_dataset = tf.data.Dataset.from_generator(lambda: train_data, output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])), output_types=('float64', 'float64'))
    # train_dataset = train_dataset.batch(1)
    # print("train dataset loaded")

    # test_data = pt.prepare_dataset(images_test, densities_test) # returns tuples
    # test_dataset = tf.data.Dataset.from_generator(lambda: test_data, output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])), output_types=('float64', 'float64'))
    # test_dataset = test_dataset.batch(1)
    # print("test dataset loaded")


    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        # metrics=["sparse_categorical_accuracy"])
    )
    # if ARGS.evaluate:
    # else:
    train(model, train_dataset, test_dataset, checkpoint_path)
    test(model, test_dataset)


if __name__ == '__main__':
    main()


# # VGG Model modified for 3 way classification
# vgg16 = VGG16(weights=None, include_top=True)

# #Add a layer where input is the output of the  second last layer 
# x = Dense(3, activation='softmax', name='predictions')(vgg16.layers[-2].output)

# #Then create the corresponding model 
# my_model = Model(input=vgg16.input, output=x)
# my_model.summary()

# # from the article: The classifier is trained on the labels of multichotomy generated from differential training. 
# # shouldn't I have the weights/checkpoints saved somewhere?

# num_epochs = 100

# images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
# densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")

# train_dataset = new_pretraining.prepare_dataset(images, densities)

# models = [pt.R1Model(), pt.R2Model(), pt.R3Model()]

# # Tc Epochs
# for t in range(num_epochs):
#     # generate labels for training switch

#     l_best = []
#     for i in range(len(train_dataset)):
#         minval_holder = []
#         for model in enumerate(models):
#             minval_holder.append(np.subtract(model(i) - densities(i))

#         l_best.append(min(minval_holder))


# # Train switch classifier。。。this is copied from the github as is. 
# num_epochs = 1
# for epoch in range(num_epochs):
#     avg_pc_loss = 0.0
#     random.shuffle(train_data)
#     for i, (X, Y) in enumerate(train_data):
#         pc_loss = train_funcs[0](X, Y, lr)
#         avg_pc_loss += pc_loss
#         # if i % 500 == 0:
#         #     log(log_fp, 'iter: %d, pc_loss: %f, avg_pc_loss: %f' % \
#         #                     (i, pc_loss, avg_pc_loss / (i + 1)))
#     avg_pc_loss /= (i + 1)
    # log(log_fp, 'done; avg_pc_Loss: %.12f' % (avg_pc_loss))




########################################## if we want to change the last few layers. 
# # #Get back the convolutional part of a VGG network trained on ImageNet
# model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
# model_vgg16_conv.summary()

# #Create your own input format (here 3x200x200)
# input = Input(shape=(3,200,200),name = 'image_input')

# #Use the generated model 
# output_vgg16_conv = model_vgg16_conv(input)

# #Add the fully-connected layers 
# x = Flatten(name='flatten')(output_vgg16_conv)
# # x = Dense(4096, activation='relu', name='fc1')(x)
# # x = Dense(4096, activation='relu', name='fc2')(x)
# x = Dense(3, activation='softmax', name='predictions')(x)

# #Create your own model 
# my_model = Model(input=input, output=x)

# #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
# my_model.summary()


########################################## if we don't want to change the last few layers. 
###why is the weights none?
# Generate a model with all layers (with top)


# images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
# # iamges = images[0][0]
# densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")
# densities = densities[0]

# my_model.compile(loss = "binary_crossentropy", 
#                     optimizer = SGD(lr=1e-4, momentum=0.9), 
#                     metrics=["accuracy"])

# history = my_model.fit(images,densities, batch_size=batch, epochs=epochs, 
#                     verbose=1,validation_split=0.2, shuffle=True)

# my_model(images, densities)
# hist = my_model.fit()

# models = [pretraining.R1Model, ... ]

# #how to 
# for image in images:
#     for model in models:
#         smallest one gets saved. 

#     #training switch. 
#     #switch differential returns the best R. 