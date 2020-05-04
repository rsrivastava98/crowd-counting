import numpy as np 
from skimage.transform import resize, rescale
a = np.arange(100).reshape((5,5,4))
print(a)
b = np.sum(a, axis=(1,2))
print(b)
# b = resize(a, (2,2), anti_aliasing=False)
# c = rescale(a, 0.25, anti_aliasing=False)
# print(b)
# print(c)
# c = 25.0/2.0
# d = np.floor(c)
# print(c,d)