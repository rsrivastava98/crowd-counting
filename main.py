import numpy as np
from skimage import io
import scipy.io as sco
from scipy import misc
import os
from tensorflow.keras.preprocessing import image
import preprocessing
import sys
from joblib import Parallel, delayed
import new_pretraining
import differential
import new_coupled_training


__DATASET_ROOT = "data/shanghaitech_h5_empty/ShanghaiTech/"


def main():
    a_train, a_test, b_train, b_test = preprocessing.generate_shanghaitech_path(__DATASET_ROOT)

    print("Generating Density Maps...")

    output_name = "ShanghaiTech_PartA_Train/"
    Parallel(n_jobs=4)(delayed(preprocessing.density_wrapper)(i, output_name) for i in a_train)

    output_name = "ShanghaiTech_PartA_Test/"
    Parallel(n_jobs=5)(delayed(preprocessing.density_wrapper)(i, output_name) for i in a_test)

    # output_name = "ShanghaiTech_PartB_Train/"
    # Parallel(n_jobs=4)(delayed(preprocessing.density_wrapper)(i, output_name) for i in b_train)

    # output_name = "ShanghaiTech_PartB_Test/"
    # Parallel(n_jobs=4)(delayed(preprocessing.density_wrapper)(i, output_name) for i in b_test)

    print("Pretraining...")
    new_pretraining.main()

    print("Differential Training...")
    differential.main()

    print("Coupled Training...")
    new_coupled_training.train()


if __name__ == '__main__':
    main()
