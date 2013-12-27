#!/usr/bin/env python

import numpy as np
import scipy as sp
from ipcv import gradient_orientation
from ipcv.util import imsave


def visualize_go():
    ''' Visualize the gradient orientation responses. '''
    img = sp.misc.imread('data/patterns.png', flatten=True)
    go, go_m = gradient_orientation(img, 2.5, signed=True, fft=False)
    imsave('go/patterns-go.png', go)
    imsave('go/patterns-go_weighted.png', go*go_m)
    go, go_m = gradient_orientation(img, 2.5, signed=False, fft=False)
    imsave('go/patterns-go_unsigned.png', go)


def visualize_go_fiducial_orientation():
    ''' Visualize the shape index orientation with a fiducial coordinate
        system.'''
    img = sp.misc.imread('data/rings.png', flatten=True)
    go, go_m = gradient_orientation(img, 2.5, signed=True, fft=False)
    # No fiducial coordinate system.
    imsave('go/rings-go.png', go)

    # Let each pixel have its own coordinate system where the origin is
    # the image center and the first axis is the vector from the origin to
    # the pixel.
    h, w = img.shape[:2]
    y = np.linspace(-h/2., h/2., h)
    x = np.linspace(-w/2., w/2., w)
    xv, yv = np.meshgrid(x, y)
    offsets = np.arctan2(yv, xv)
    go = np.mod(go-offsets, 2*np.pi)
    imsave('go/rings-go-fiducial.png', go)
    imsave('go/rings-go-fiducial_weighted.png', go*go_m)

if __name__ == '__main__':
    visualize_go()
    visualize_go_fiducial_orientation()
