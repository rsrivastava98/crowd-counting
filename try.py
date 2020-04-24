from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
from PIL import Image
from matplotlib.image import imread
# matplotlib.use("TkAgg") 



# io.imshow(data)
# plt.show()

# import os
# import glob

# path, dirs, files = next(os.walk("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5"))
# file_count = len(files)

# filenames = [img for img in glob.glob("ShanghaiTech_PartA_Test/part_A/test_data/ground-truth-h5/*.h5")]

# filenames.sort() # ADD THIS LINE
# densities = []
# for img in filenames:
#     hf = h5py.File(img, 'r')
#     data = hf.get('density')[()]

#     shape = data.shape
#     px = int(shape[0] / 3)
#     py = int(shape[1] / 3)
    
#     for i in range(3):
#         for j in range(3):
#             n = data [px*i:px*(i+1), py*j:py*(j+1)]
#             densities.append(n)

# densities = np.array(densities)



import scipy
import os
import glob
from skimage import color
from skimage import io

path, dirs, files = next(os.walk("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images"))
file_count = len(files)
# print(file_count)

filenames = [img for img in glob.glob("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images/*.jpg")]

filenames.sort()

images = []
for img in filenames:
    # print(img)
    data = color.rgb2gray(io.imread(img))
    # print(data)
    

    shape = data.shape
    px = int(shape[0] / 3)
    py = int(shape[1] / 3)
    
    for i in range(3):
        for j in range(3):
            n = data [px*i:px*(i+1), py*j:py*(j+1)-1]
            
            images.append(n)
   
    

    images = np.array(images)
    print(images)
    print(type(images))
    break




