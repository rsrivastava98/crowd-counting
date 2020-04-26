import numpy as np
from skimage import io
import scipy.io as sco
from scipy import misc
import os
from tensorflow.keras.preprocessing import image
import preprocessing
import sys
from joblib import Parallel, delayed
import pretraining


__DATASET_ROOT = "data/shanghaitech_h5_empty/ShanghaiTech/"


def main():
    a_train, a_test, b_train, b_test = preprocessing.generate_shanghaitech_path(__DATASET_ROOT)

    ### READ THIS! So there's a chance your computer will overheat, so I recommend uncommenting
    # one block at a time (leave the other 3 blocks commented out) because it will take about 20-30 minutes/block...

    output_name = "ShanghaiTech_PartA_Train/"
    Parallel(n_jobs=4)(delayed(preprocessing.density_wrapper)(i, output_name) for i in a_train)

    output_name = "ShanghaiTech_PartA_Test/"
    Parallel(n_jobs=4)(delayed(preprocessing.density_wrapper)(i, output_name) for i in a_test)

    # output_name = "ShanghaiTech_PartB_Train/"
    # Parallel(n_jobs=4)(delayed(preprocessing.density_wrapper)(i, output_name) for i in b_train)

    # output_name = "ShanghaiTech_PartB_Test/"
    # Parallel(n_jobs=4)(delayed(preprocessing.density_wrapper)(i, output_name) for i in b_test)

    #densities = preprocessing.density_patches("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5")
    #images = preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images")

    # pretraining.main((images, densities))

    # pretraining.main(np.array(images, densities))



if __name__ == '__main__':
    main()
