import preprocessing
import new_pretraining 
import numpy as np
import tensorflow as tf
from tensorflow import keras

def differential_train(train_data, test_data, networks):

    num_epochs = 1

    for model in networks:
         model.compile(
            optimizer= tf.keras.optimizers.SGD(learning_rate = 0.0005, momentum = 0.9),
            loss=model.loss_fn
            )

    min_mae = np.zeros(num_epochs)
    num_nets = 3

    for epoch in range(num_epochs):
        print("epoch" + str(epoch))
        switch_stat = np.zeros(num_nets)
        for i, example in enumerate(train_data):
            image = example[0]
            density = example[1]
            net_losses = np.zeros(num_nets)
            im = image.reshape((1, image.shape[0], image.shape[1], 1))
            dens = density.reshape((1, density.shape[0], density.shape[1], 1))

            for j, model in enumerate(networks):
                y_pred = model.predict(im)
                net_losses[j] = np.abs(np.sum(y_pred) - np.sum(density))
            
            y_pc = np.argmin(net_losses)
            model = networks[y_pc]

            x = tf.constant(im, dtype='float32')
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_weights)
                loss = model.loss(model.call(x), dens)
                #loss = np.abs(np.sum(model.call(im)) - np.sum(density))

            grads = tape.gradient(loss, model.trainable_weights)

            networks[y_pc].optimizer.apply_gradients(zip(grads, networks[y_pc].trainable_weights))

            switch_stat[y_pc] += 1
        
        print(switch_stat)

        min_mae[epoch] = calc_min_mae(test_data, networks)  # function be written based on test_cnn

        #save weights
    
    print(min_mae)


    for i, network in enumerate(networks):
        network.save_weights('./differential_checkpoints/r'+str(i+1)+'model')


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
        

def main():

    train_images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
    train_densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")

    train_dataset = new_pretraining.prepare_dataset(train_images, train_densities)

    test_images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images")
    test_densities = preprocessing.density_patches("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5")

    test_dataset = new_pretraining.prepare_dataset(test_images, test_densities)

    model1 = new_pretraining.R1Model()
    model1(tf.keras.Input(shape=(None, None, 1)))

    model2 = new_pretraining.R2Model()
    model2(tf.keras.Input(shape=(None, None, 1)))

    model3 = new_pretraining.R3Model()
    model3(tf.keras.Input(shape=(None, None, 1)))

    networks = [model1, model2, model3]

    checkpoint_paths = ["r1_checkpoints/weights.h5", "r2_checkpoints/weights.h5", "r3_checkpoints/weights.h5"] # model weight paths for all models REQUIRED TO LOAD BEST WEIGHTS FROM PRETRAINING
    for i in range(len(networks)):
        networks[i].load_weights(checkpoint_paths[i])
    
    differential_train(train_dataset, test_dataset, networks)



if __name__ == '__main__':
    main()