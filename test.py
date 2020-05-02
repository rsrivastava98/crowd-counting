import numpy as np 
from skimage.transform import resize, rescale
a = np.arange(64).reshape((8,8))
print(a)
b = resize(a, (2,2), anti_aliasing=False)
c = rescale(a, 0.25, anti_aliasing=False)
print(b)
print(c)
# c = 25.0/2.0
# d = np.floor(c)
# print(c,d)