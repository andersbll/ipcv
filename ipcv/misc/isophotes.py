import numpy as np


def isophotes(img, n, limits, scale, smoothing_fun='gaussian'):
    """Generate soft isophote images.

    Generate n soft isophote images with equally spaced isophote lines between
    limits[0] and limits[1]. The isophote images are soft because the
    contributions to each isophote line are smoothed with a Gaussian function.

    Args:
        img: Grayscale image as a (p,q) array.
        n: The number of isophote images to be created.
        limits: A two-tuple containing the lower and upper limit of the
            isophote lines.
        scale: The scale adjusts the width of the Gaussian kernel used for
            smoothing bin contributions.
        smoothing_fun: 'gaussian' selects Gaussian smoothing. 'von_mises'
            selects the Von Mises distribution (aka. circular normal
            distribution). Von mises is useful for image intensities of
            periodic nature.

    Returns:
        Isophote images of img stored in a (n, p, q) array.
    """
    iso = np.empty((n,) + img.shape)
    limit_size = limits[1]-limits[0]
    if smoothing_fun == 'gaussian':
        step = limit_size/float(n)
        centers = np.linspace(limits[0]+step*.5, limits[1]-step*.5, n)
        for i, c in enumerate(centers):
            iso[i, :, :] = np.exp(-(img-c)**2/(2*scale**2))
    elif smoothing_fun == 'von_mises':
        img *= 2*np.pi/limit_size
        step = 2*np.pi/float(n)
        centers = np.linspace(-np.pi+step*.5, np.pi-step*.5, n)
        kappa = 1/(scale**2)
        for i, c in enumerate(centers):
            iso[i, :, :] = np.exp(kappa * np.cos(img-c))
    else:
        raise ValueError('Invalid smoothing function.')
    return iso
