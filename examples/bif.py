#!/usr/bin/env python

import scipy as sp
from ipcv import bif_response, bif_colors
from ipcv.util import imsave


def visualize_bif():
    ''' Visualize the shape index responses. '''
    img = sp.misc.imread('data/camera.png', flatten=True)
    bif_img = bif_colors(bif_response(img, 2, eps=0.02))
    imsave('bif/lena-bif-sigma2.png', bif_img)
    bif_img = bif_colors(bif_response(img, 4, eps=0.02))
    imsave('bif/lena-bif-sigma4.png', bif_img)
    bif_img = bif_colors(bif_response(img, 8, eps=0.02))
    imsave('bif/lena-bif-sigma8.png', bif_img)


if __name__ == '__main__':
    visualize_bif()
