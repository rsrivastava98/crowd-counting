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

def train_switch():

    #TODO: create label set for training switch classifier

    #TODO: get label for every sample
    for i, (X, Y) in enumerate(train_dataset):
        #for i in test_fn 
            #y_prediction = test_fn(x, y) -- call model.fit()
            #calculate loss as absolute difference
        #y_pc = argmin of losses
        #append to eq_data

    #TODO: equalize number of samples across classes 

    #TODO: train switch classifer
    num_epochs = 1
    for epoch in range(num_epochs):
        avg_pc_loss = 0.0
        #shuffle train data
        for i, (X, Y) in enumerate(train_dataset):
            pc_loss = train_funcs[0](X, Y, lr)
            avg_pc_loss += pc_loss
            # if i % 500 == 0: calculate average
        print(avg_pc_loss)

def train_switched_differential():

    #for epoch
    num_epochs = 1
    num_nets = 3
    for epoch in range(num_epochs):
        switch_stat = np.zeros(num_nets)

        #for image in training data
        for i, example in enumerate(train_data):
            image = example[0]
            density = example[1]
        
            #TODO: run switch classifier to get label
            label = run_funcs[0](X) #this line gets the label from switch
            y_pc = np.argmax(label, axis = 1)[0] #this line stores which regressor is chosen
    
            #TODO: backpropagate regressor suggested by classifier
            model_chosen = networks[y_pc + 1] #gets the chosen regressor
            #compute loss for chosen regressor
            #backpropogate

def coupled_train(train_data, test_data, networks):

    num_epochs = 30
    min_mae = np.zeros(num_epochs)
    for each epoch in range(num_epochs):
        train_switch()
        train_switched_differential()

        min_mae[epoch] = calc_min_mae(test_data, networks)
    
    print(min_mae)

    for i, network in enumerate(networks):
        network.save_weights('./coupled_checkpoints/r'+(i+1)+'model')

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

    for i, (X, Y) in enumerate(test_data):
        image = X
        density = Y
        reshaped_image = image.reshape((1, image.shape[0], image.shape[1], 1))
        reshaped_density = density.reshape((1, density.shape[0], density.shape[1], 1))
        x = tf.constant(reshaped_image, dtype='float32')
        for j, network in enumerate(networks):
            y_pred = network.call(x)
            patch_counts_sub[j, 0] =np.sum(y_pred)
            patch_counts_sub[j, 1] = np.sum(reshaped_density)
            patch_counts_total[j+1, 0] = patch_counts_sub[j, 0]
            patch_counts_total[j+1, 1] = patch_counts_sub[j, 1]
            counts[j] = np.abs(patch_counts_sub[j, 0] - patch_counts_sub[j, 1])
            loss[j] = networks[j].loss(y_pred, reshaped_density)
        
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
    
    print(num_images)
    total_losses /= len(test_data)
    total_counts /= len(test_data)
    #switch_stat /= len(test_data)
    #pc_switch_stat /= i
    #pc_switch_error /= i
    mae /= num_images

    print(mae)
    print(total_counts)
    print(total_losses)

    return mae[-1]

#main train function
def train():

    # load and set up data sets
    train_images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
    train_densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")

    train_dataset = new_pretraining.prepare_dataset(train_images, train_densities)

    test_images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images")
    test_densities = preprocessing.density_patches("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5")

    test_dataset = new_pretraining.prepare_dataset(test_images, test_densities)

    # get weights from differential
    #TODO: still need to add file name
    checkpoint_paths = ["./differential_checkpoints/r1model/",
                        "./differential_checkpoints/r2model/",
                        "./differential_checkpoints/r3model/"]

    # make networks list
    model1 = new_pretraining.R1Model()
    model1(tf.keras.Input(shape=(None, None, 1)))

    model2 = new_pretraining.R2Model()
    model2(tf.keras.Input(shape=(None, None, 1)))

    model3 = new_pretraining.R3Model()
    model3(tf.keras.Input(shape=(None, None, 1)))

    networks = [model1, model2, model3]

    # load checkpoints into models
    for i in range(len(networks)):
        networks[i].load_weights(checkpoint_paths[i])

    #TODO: Add VGG weights loading
    
    # train_coupled    
    coupled_train(train_dataset, test_dataset, networks) 

