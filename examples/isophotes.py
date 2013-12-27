#!/usr/bin/env python

import numpy as np
import scipy as sp
from ipcv import gradient_orientation
from ipcv.misc import isophotes
from ipcv.util import imsave


def visualize():
    ''' Visualize the shape index responses. '''
    img = sp.misc.imread('data/patterns.png', flatten=True)
    go, go_m = gradient_orientation(img, 3)
    iso_go = isophotes(go, 5, (-np.pi, np.pi), .5, 'gaussian')
    imsave('isophotes/patterns-go.png', go*go_m)
    for i in range(iso_go.shape[0]):
        imsave('isophotes/patterns-go_iso_%i_gaussian.png'%i,
               iso_go[i, ...]*go_m)
    iso_go = isophotes(go, 5, (-np.pi, np.pi), .5, 'von_mises')
    for i in range(iso_go.shape[0]):
        imsave('isophotes/patterns-go_iso_%i_von_mises.png'%i,
               iso_go[i, ...]*go_m)


if __name__ == '__main__':
    visualize()
