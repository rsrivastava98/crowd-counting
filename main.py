import numpy as np
from skimage import io 
import scipy.io as sco
from scipy import misc
import os

def main():
    train_images_path = "data/ShanghaiTech/part_A/train_data/"
    test_images_path = "data/ShanghaiTech/part_A/test_data/"

    img_file_list = []
    ground_truth_list = []

    for root, _, files in os.walk(train_images_path):
        for name in files:   
            if name.endswith(".jpg"):
                img_file_list.append(os.path.join(root, name))
            if name.endswith(".mat"):
                ground_truth_list.append(os.path.join(root, name))
      
    
    for img_path in img_file_list:
        img = io.imread(img_path)

    for ground_truth in ground_truth_list:
        points = sco.loadmat(ground_truth)
        print(points)
        

if __name__ == '__main__':
    main()


