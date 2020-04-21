import numpy as np
from skimage import io
import scipy.io as sco
from scipy import misc
import os
from tensorflow.keras.preprocessing import image
import preprocessing


def main():
    train_images_path = "data/ShanghaiTech/part_A/train_data/"
    test_images_path = "data/ShanghaiTech/part_A/test_data/"

    img_file_list = []
    
    for root, _, files in os.walk(train_images_path):
        for name in files:
            if name.endswith(".jpg"):
                img_file_list.append(os.path.join(root, name))


    for i in range(1):
        imgfile = image.load_img(img_file_list[i])
        mat_file = img_file_list[i].replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
        img = image.img_to_array(imgfile)
        points = sco.loadmat(mat_file)
        arr = points['image_info'][0][0][0][0][0]
        print(img_file_list[i], mat_file)
        preprocessing.generate_density_map(img, arr)



if __name__ == '__main__':
    main()
