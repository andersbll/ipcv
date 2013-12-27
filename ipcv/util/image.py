import os
import numpy as np
import scipy as sp


def stretch_intensity(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img)+1e-12
    return img


def imsave(path, img, stretch=True):
    if stretch:
        img = (255*stretch_intensity(img)).astype(np.uint8)
    dirpath = os.path.dirname(path)
    if len(dirpath) > 0 and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    sp.misc.imsave(path, img)
