import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np
import preprocessing
import sys
import new_pretraining
import random
import vgg_model
import hyperparameters as hp
from keras.utils.conv_utils import convert_kernel 

def train_switch(train_data, test_data, networks):
    train_images, _ = train_data
    test_images, _ = test_data
    train_labels = np.ones(20)
    test_labels = np.ones(20)

    train_images = train_images[0:20]
    test_images = test_images[0:20]


    #TODO: create label set for training switch classifier

    #TODO: get label for every sample
    # for i, (X, Y) in enumerate(train_dataset):
        #for i in test_fn 
            #y_prediction = test_fn(x, y) -- call model.fit()
            #calculate loss as absolute difference
        #y_pc = argmin of losses
        #append to eq_data

    #TODO: equalize number of samples across classes 

    #TODO: train switch classifer
    num_epochs = 1
    # for epoch in range(num_epochs):
        # avg_pc_loss = 0.0
        # shuffle train data
        # for image in enumerate(train_images):

    checkpoint_path = "training_1/cp.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Train the model with the new callback
    networks[0].fit(train_images, 
        train_labels,  
        epochs=10,
        validation_data=(test_images,test_labels),
        callbacks=[cp_callback]) 

            # pc_loss = train_funcs[0](X, Y, lr)
            # avg_pc_loss += pc_loss
            # if i % 500 == 0: calculate average
        # print(avg_pc_loss)

def train_switched_differential(train_data, test_data, networks):

    #for epoch
    num_epochs = 30
    num_nets = 3
    for epoch in range(num_epochs):
        switch_stat = np.zeros(num_nets)

        #for image in training data
        for i, example in enumerate(train_data):
            image = example[0]
            density = example[1]

            im = image.reshape((1, image.shape[0], image.shape[1], 1))
            dens = density.reshape((1, density.shape[0], density.shape[1], 1))

            # run switch classifier to get label
            label = networks[0].predict(im) #this line gets the label from switch
            y_pc = np.argmax(label, axis = 1)[0] #this line stores which regressor is chosen

            #backpropagate regressor suggested by classifier
            model_chosen = networks[y_pc + 1] #gets the chosen regressor

            #backpropogate from differential
            x = tf.constant(im, dtype='float32')
            with tf.GradientTape() as tape:
                tape.watch(model_chosen.trainable_weights)
                loss = model_chosen.loss_fn(model_chosen.call(x), dens)
                #loss = np.abs(np.sum(model.call(im)) - np.sum(density))

            grads = tape.gradient(loss, model_chosen.trainable_weights)

            networks[y_pc+1].optimizer.apply_gradients(zip(grads, networks[y_pc+1].trainable_weights))

            switch_stat[y_pc] += 1

def coupled_train(train_data, test_data, networks):

    num_epochs = 1

    networks[0].compile(
        optimizer=networks[0].optimizer,
        loss=networks[0].loss_fn,
        metrics=["sparse_categorical_accuracy"])

    for i in range(1,3):
        networks[i].compile(
            optimizer= tf.keras.optimizers.SGD(learning_rate = 0.0005, momentum = 0.9),
            loss=networks[i].loss_fn
            )

    min_mae = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        train_switch(train_data, test_data, networks)
        break
        train_switched_differential(train_data, test_data, networks)

        min_mae[epoch] = calc_min_mae(test_data, networks)
        print(min_mae)
    


    for i, network in enumerate(networks):
        network.save_weights('./coupled_checkpoints/r'+str((i+1))+'model.h5')

#TODO: add switch stuff
def calc_min_mae(test_data, networks):

    patch_counts_total = np.zeros((5, 2))
    patch_counts_sub = np.zeros((3,2))
    loss = np.zeros(3)
    counts = np.zeros(3)
    mae = np.zeros(5)
    total_losses = np.zeros(5)
    total_counts = np.zeros(5)
    patchct = 0
    num_images = 0

    pc_switch_stat = np.zeros(3)
    calc_switch_stat = np.zeros(3)
    pc_switch_error = 0.0

    for i, (X, Y) in enumerate(test_data):
        image = X
        density = Y
        reshaped_image = image.reshape((1, image.shape[0], image.shape[1], 1))
        reshaped_density = density.reshape((1, density.shape[0], density.shape[1], 1))
        x = tf.constant(reshaped_image, dtype='float32')
        networks_noswitch = networks[1:]
        for j, network in enumerate(networks_noswitch):
            y_pred = network.call(x)
            patch_counts_sub[j, 0] =np.sum(y_pred)
            patch_counts_sub[j, 1] = np.sum(reshaped_density)
            patch_counts_total[j+1, 0] = patch_counts_sub[j, 0]
            patch_counts_total[j+1, 1] = patch_counts_sub[j, 1]
            counts[j] = np.abs(patch_counts_sub[j, 0] - patch_counts_sub[j, 1])
            loss[j] = networks_noswitch[j].loss_fn(y_pred, reshaped_density)
        
        y_pc = np.argmin(counts)
        total_losses[-1] += loss[y_pc]
        total_counts[-1] += counts[y_pc]

        patch_counts_total[-1, 0] += patch_counts_sub[y_pc, 0]
        patch_counts_total[-1, 1] += patch_counts_sub[y_pc, 1]

        #TODO: add switch stuff here

        total_losses[1: -1] += loss
        total_counts[1: -1] += counts
        patchct+= 1

        # Compute MAE
        if patchct >= 9:
            patchct = 0
            mae += np.abs(patch_counts_total[:, 0] - patch_counts_total[:, 1])
            patch_counts_total[:, :] = 0
            num_images += 1
    
    # print(num_images)
    total_losses /= len(test_data)
    total_counts /= len(test_data)
    #switch_stat /= len(test_data)
    #pc_switch_stat /= i
    #pc_switch_error /= i
    mae /= num_images

    # print(mae)
    # print(total_counts)
    # print(total_losses)

    return mae[-1]

#main train function
def train():

    # load and set up data sets
    train_images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images", grayscale=False)
    train_densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")
    
    train_images = train_images[0:20]
    train_densities = train_densities[0:20]
    
    train_data = new_pretraining.prepare_dataset(train_images, train_densities, grayscale=False)
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data, output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])), output_types=('float64', 'float64'))
    train_dataset = train_dataset.batch(1)

    # print(train_dataset[0].shape)
    # print(train_dataset[0][0].shape)
    # print(train_dataset[0][0])

    test_images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images", grayscale=False)
    test_densities = preprocessing.density_patches("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5")

    test_images = train_images[0:20]
    test_densities = train_densities[0:20]

    test_data = new_pretraining.prepare_dataset(test_images, test_densities, grayscale=False)
    test_dataset = tf.data.Dataset.from_generator(lambda: test_data, output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])), output_types=('float64', 'float64'))
    test_dataset = test_dataset.batch(1)

    print("loaded data")

    # get weights from differential
    #TODO: need to fix vgg weights
    checkpoint_paths = ["./vgg16_weights_4.h5",
                        "./r1_checkpoints/weights.h5",
                        "./r2_checkpoints/weights.h5",
                        "./r3_checkpoints/weights.h5"]

    # make networks list
    model1 = new_pretraining.R1Model()
    model1(tf.keras.Input(shape=(None, None, 1)))

    model2 = new_pretraining.R2Model()
    model2(tf.keras.Input(shape=(None, None, 1)))

    model3 = new_pretraining.R3Model()
    model3(tf.keras.Input(shape=(None, None, 1)))

    # switch_model = VGG16(weights=None, include_top=False, input_shape=(None, None, 1), classes=3)
    switch_model = vgg_model.VGGModel()
    switch_model(tf.keras.Input(shape=(None, None, 3)))

    networks = [switch_model, model1, model2, model3]

    # load checkpoints into models
    # for i in range(1,len(networks)):
    #     networks[i].load_weights(checkpoint_paths[i])
    print("loaded checkpoints")

    # train_coupled
    networks[0].load_weights(checkpoint_paths[0], by_name=True)
    train_switch(train_dataset, test_dataset, networks)    
    # coupled_train(train_dataset, test_dataset, networks) 

if __name__ == '__main__':
    train()