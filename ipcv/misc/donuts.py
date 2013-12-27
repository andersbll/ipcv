import numpy as np


def donut(shape, radius, width, distribution='gaussian'):
    '''Generate a 2D Gaussian window of the given shape. width specifies the
    size of the Gaussian. radius specifies the distance to origo such that
    the window becomes a ring.'''
    if not distribution in ['gaussian', 'lognormal']:
        raise ValueError('Invalid distribution function specified.')
    h, w = shape
    y = np.linspace(-h/2., h/2., h)
    x = np.linspace(-w/2., w/2., w)
    xv, yv = np.meshgrid(x, y)
    if distribution == 'lognormal' and radius > 0:
        mean = radius
        var = width**2
        mu = np.log(mean**2 / np.sqrt(var + mean**2))
        sigma = np.sqrt(np.log(var/mean**2 + 1))
        d = np.sqrt(xv**2+yv**2) + 1e-5
        return 1/(d*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(d)-mu)**2
                                                   / (2*sigma**2))
    else:
        sigma = width
        mu = radius
        return np.exp(-(np.sqrt(xv**2+yv**2) - mu)**2/(2*sigma**2))


def donuts(shape, n_donuts, radius_max, width_min, width_ratio=1.0,
           distribution='gaussian'):
    radii = np.linspace(0, radius_max, n_donuts)
    widths = [float(width_min)*width_ratio**i for i in range(n_donuts)]
    weights = [donut(shape, r, w, distribution)
               for (r, w) in zip(radii, widths)]
    weights = [w/np.sum(w) for w in weights]
    return weights
