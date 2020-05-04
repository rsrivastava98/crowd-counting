import preprocessing
import new_pretraining 
import numpy as np
import tensorflow as tf

def differential_train(train_data, networks):


    num_epochs = 100 

    min_mae = np.zeros(num_epochs)
    num_nets = 3

    for epoch in range(num_epochs):
        switch_stat = np.zeros(num_nets)
        for i, example in enumerate(train_data):
            image = example[0]
            density = example[1]
            net_losses = np.zeros(num_nets)

            for j, model in enumerate(networks):
                y_pred = model.call(image)
                net_losses[j] = np.abs(np.sum(y_pred) - np.sum(density))
            
            y_pc = np.argmin(net_losses)

            networks[y_pc].compile(
                'sgd',
                loss=tf.keras.losses.MeanSquaredError()
            )

            dataset = tf.data.Dataset.from_generator(lambda: example, output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])), output_types=('float64', 'float64'))
            dataset = dataset.batch(1)

            networks[y_pc].fit(x = dataset,
                    epochs= 50,
                    batch_size= None)

            switch_stat[y_pc] += 1
        
        print(switch_stat)

        # min_mae[epoch] = calc_min_mae()  # function be written based on test_cnn

        #save weights


def main():

    images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images")
    densities = preprocessing.density_patches("ShanghaiTech_PartA_Train/part_A/train_data/ground-truth-h5")

    networks = [new_pretraining.R1Model(), new_pretraining.R2Model(), new_pretraining.R3Model()]

    train_dataset = new_pretraining.prepare_dataset(images, densities)

    differential_train(train_dataset, networks)

    checkpoint_paths = ["r1_checkpoints/weights.e01.data-00001-of-00002", "r2_checkpoints/weights.e01.data-00001-of-00002", "r3_checkpoints/weights.e01.data-00001-of-00002"] # model weight paths for all models REQUIRED TO LOAD BEST WEIGHTS FROM PRETRAINING
    for i, network in enumerate(networks):
        network.load_weights(checkpoint_paths[i])


if __name__ == '__main__':
    main()