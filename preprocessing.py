import numpy as np
import os
from skimage import io
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from scipy.io import loadmat
import glob
import h5py
import time
# from sklearn.externals.joblib import Parallel, delayed
import sys
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from matplotlib.image import imread
from skimage import color


__DATASET_ROOT = "data/shanghaitech_h5_empty/ShanghaiTech/"



def generate_shanghaitech_path(root):
    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B/test_data', 'images')

    img_paths_a_train, img_paths_a_test, img_paths_b_train, img_paths_b_test = [], [], [], []

    for roots, _, files in os.walk(part_A_train):
        for name in files:
            if name.endswith(".jpg"):
                img_paths_a_train.append(os.path.join(roots, name))

    for roots, _, files in os.walk(part_A_test):
        for name in files:
            if name.endswith(".jpg"):
                img_paths_a_test.append(os.path.join(roots, name))

    for roots, _, files in os.walk(part_B_train):
        for name in files:
            if name.endswith(".jpg"):
                img_paths_b_train.append(os.path.join(roots, name))

    for roots, _, files in os.walk(part_B_test):
        for name in files:
            if name.endswith(".jpg"):
                img_paths_b_test.append(os.path.join(roots, name))

    return img_paths_a_train, img_paths_a_test, img_paths_b_train, img_paths_b_test



def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.sum(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    neigh = NearestNeighbors(n_neighbors=4)
    neigh.fit(pts)
    distances = neigh.kneighbors(pts)[0]

    # print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    # print('done.')
    return density

def generate_density_map(img, gt, img_path, h5_path):
    gt_loc = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            gt_loc[int(gt[i][1]), int(gt[i][0])] = 1
    gt_loc = gaussian_filter_density(gt_loc)

    output_path = img_path.replace(__DATASET_ROOT, h5_path).replace('.jpg', '.h5').replace('images','ground-truth-h5')
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print("output", output_path)

    # io.imshow(gt_loc)
    # plt.show()
    sys.stdout.flush()
    with h5py.File(output_path, 'w') as hf:
        hf['density'] = gt_loc

def density_wrapper(img_path, output_path):
    imgfile = image.load_img(img_path)
    mat_file = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
    img = image.img_to_array(imgfile)
    points = loadmat(mat_file)
    arr = points['image_info'][0][0][0][0][0]
    print(img_path, mat_file)
    generate_density_map(img, arr, img_path, output_path)

def density_patches(data_path):
    path, dirs, files = next(os.walk(data_path))
    file_count = len(files)

    filenames = [img for img in glob.glob(data_path + "/*.h5")]

    filenames.sort() 
    densities = []
    for img in filenames:
        hf = h5py.File(img, 'r')
        data = hf.get('density')[()]

        shape = data.shape
        px = int(shape[0] / 3)
        py = int(shape[1] / 3)
        
        for i in range(3):
            for j in range(3):
                n = data [px*i:px*(i+1), py*j:py*(j+1)-1]
                densities.append(n)
    # densities = np.array(densities)
    return densities

def image_patches(data_path, color=False):
    path, dirs, files = next(os.walk(data_path))
    file_count = len(files)

    filenames = [img for img in glob.glob(data_path + "/*.jpg")]

    filenames.sort()

    images = []
    for img in filenames:
        
        if color==True:
            data = io.imread(img)
        else:
            data = color.rgb2gray(io.imread(img))
        
        shape = data.shape
        px = int(shape[0] / 3)
        py = int(shape[1] / 3)
        
        for i in range(3):
            for j in range(3):
                n = data [px*i:px*(i+1), py*j:py*(j+1)-1]
                images.append(n)

    # images = np.array(images)
    return images
