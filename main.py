import numpy as np
from skimage import io
import scipy.io as sco
from scipy import misc
import os
from tensorflow.keras.preprocessing import image
import preprocessing
import sys
from joblib import Parallel, delayed


__DATASET_ROOT = "data/shanghaitech_h5_empty/ShanghaiTech/"

def density_wrapper(img_path, output_path):
    imgfile = image.load_img(img_path)
    mat_file = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
    img = image.img_to_array(imgfile)
    points = sco.loadmat(mat_file)
    arr = points['image_info'][0][0][0][0][0]
    print(img_path, mat_file)
    preprocessing.generate_density_map(img, arr, img_path, output_path)


def main():
    a_train, a_test, b_train, b_test = preprocessing.generate_shanghaitech_path(__DATASET_ROOT)

    ### READ THIS! So there's a chance your computer will overheat, so I recommend uncommenting
    # one block at a time (leave the other 3 blocks commented out) because it will take about 20-30 minutes/block...

    output_name = "ShanghaiTech_PartA_Train/"
    Parallel(n_jobs=4)(delayed(density_wrapper)(i) for i in a_train)

    # output_name = "ShanghaiTech_PartA_Test/"
    # Parallel(n_jobs=4)(delayed(density_wrapper)(i) for i in a_test)

    # output_name = "ShanghaiTech_PartB_Train/"
    # Parallel(n_jobs=4)(delayed(density_wrapper)(i, output_name) for i in b_train)

    # output_name = "ShanghaiTech_PartB_Test/"
    # Parallel(n_jobs=4)(delayed(density_wrapper)(i) for i in b_test)



if __name__ == '__main__':
    main()
