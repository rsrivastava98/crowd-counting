import numpy as np
import pandas as pd
import os
from skimage import io
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from scipy.io import loadmat
import glob
import h5py
import time
from sklearn.externals.joblib import Parallel, delayed
import sys
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.sum(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    neigh = NearestNeighbors(n_neighbors=4)
    neigh.fit(pts)
    distances = neigh.kneighbors(pts)[0]

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

def generate_density_map(img, gt):

    gt_loc = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            gt_loc[int(gt[i][1]), int(gt[i][0])] = 1
    gt_loc = gaussian_filter_density(gt_loc)

    io.imshow(gt_loc)
    plt.show()
    sys.stdout.flush()
    with h5py.File("image_density", 'w') as hf:
        hf['density'] = gt_loc