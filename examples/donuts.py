#!/usr/bin/env python

from ipcv.misc import donuts
from ipcv.util import imsave


def visualize_donuts():
    ''' Visualize donut spatial weighs. '''
    ds = donuts((100, 100), 3, 30, 8.0, 1.2)
    for i, d in enumerate(ds):
        imsave('donuts/donut%i.png'%i, d)

if __name__ == '__main__':
    visualize_donuts()
