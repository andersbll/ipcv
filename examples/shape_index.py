#!/usr/bin/env python

import numpy as np
import scipy as sp
from ipcv import shape_index
from ipcv.util import imsave


def visualize_si():
    ''' Visualize the shape index responses. '''
    img = sp.misc.imread('data/patterns.png', flatten=True)
    si, si_c, si_o, si_om = shape_index(img, 2.5, orientations=True, fft=True)
    imsave('si/patterns-si.png', si)
    imsave('si/patterns-si_c.png', si_c)
    imsave('si/patterns-si_weighted.png', si*si_c)
    imsave('si/patterns-si_o.png', si_o)
    imsave('si/patterns-si_om.png', si_om)
    imsave('si/patterns-si_o_weighted.png', si_o*si_om)


def visualize_si_orientation():
    ''' Visualize the shape index orientation. '''
    img = sp.misc.imread('data/rings.png', flatten=True)
    si, si_c, si_o, si_om = shape_index(img, 2.5, orientations=True, fft=True)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img, cmap=plt.gray())
    X, Y = np.mgrid[0:si_o.shape[0], 0:si_o.shape[1]]
    u = np.cos(si_o)*si_c/np.max(si_c)
    v = np.sin(si_o)*si_c/np.max(si_c)
    quiver_mask = si_c > 10
    step = 3
    quiver_mask = quiver_mask[::step, ::step]
    Y = (Y[::step, ::step])[quiver_mask]
    X = (X[::step, ::step])[quiver_mask]
    u = (u[::step, ::step])[quiver_mask]
    v = (v[::step, ::step])[quiver_mask]
    plt.quiver(Y, X, u, v,
               pivot='mid', scale=30,
               color='r')
    plt.savefig('si/rings-si_o.pdf')


def visualize_si_fiducial_orientation():
    ''' Visualize the shape index orientation with a fiducial coordinate
        system.'''
    img = sp.misc.imread('data/rings.png', flatten=True)
    si, si_c, si_o, si_om = shape_index(img, 2.5, orientations=True, fft=True)
    # No fiducial coordinate system.
    imsave('si/rings-si_o.png', si_o)
    imsave('si/rings-si_o_weighted.png', si_o*si_om)

    # Let each pixel have its own coordinate system where the origin is
    # the image center and the first axis is the vector from the origin to
    # the pixel.
    h, w = img.shape[:2]
    y = np.linspace(-h/2., h/2., h)
    x = np.linspace(-w/2., w/2., w)
    xv, yv = np.meshgrid(x, y)
    offsets = np.arctan(yv/(xv+1e-10))
    si_o = np.mod(si_o+offsets, np.pi)
    imsave('si/rings-si_o_fiducial.png', si_o)
    imsave('si/rings-si_o_fiducial_weighted.png', si_o*si_om)


if __name__ == '__main__':
    visualize_si()
    visualize_si_orientation()
    visualize_si_fiducial_orientation()
