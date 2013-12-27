import numpy as np
from scipy.ndimage.filters import gaussian_filter


class ScaleSpace:
    def __init__(self, img_shape, sigmas, dys, dxs):
        ''' Compute the scale-space of an image.
        Upon initialization, this class precomputes the Gaussian windows used
        to smooth images of a fixed shape to save the computations at later
        points.
        '''
        assert(len(sigmas) == len(dys) == len(dxs))
        h, w = img_shape
        g_y, g_x = np.mgrid[-.5+.5/h:.5:1./h, -.5+.5/w:.5: 1./w]
        self.filters = []
        for sigma, dy, dx in zip(sigmas, dys, dxs):
            g = np.exp(- (g_x**2 + g_y**2) * (np.pi*2*sigma)**2 / 2.)
            g = np.fft.fftshift(g)
            if dy > 0 or dx > 0:
                #TODO change list(range to np.linspace or similar
                dg_y = np.array((list(range(0, h//2))+list(range(-h//2, 0))),
                                dtype=float, ndmin=2) / h
                dg_x = np.array((list(range(0, w//2))+list(range(-w//2, 0))),
                                dtype=float, ndmin=2) / w
                dg = (dg_y.T**dy) * (dg_x**dx) * (1j*2*np.pi)**(dy + dx)
                g = np.multiply(g, dg)
            self.filters.append(g)

    def compute_f(self, img_f):
        ''' Compute the scale space of an image in the fourier domain.'''
        return [np.multiply(img_f, f) for f in self.filters]

    def compute(self, img):
        ''' Compute the scale space of an image.'''
        img_f = np.fft.fft2(img)
        return [np.fft.ifft2(np.multiply(img_f, f)).real for f in self.filters]


def scalespace(img, sigma, order=(0, 0)):
    '''Compute the scale-space of an image. sigma is the scale parameter. dx
    and dy specify the differentiation order along the x and y axis
    respectively.'''
    ss = ScaleSpace(img.shape, [sigma], [order[0]], [order[1]])
    return ss.compute(img)[0]


def gradient_orientation(img, scale, signed=True, fft=False):
    '''Calculate gradient orientations at scale sigma.'''
    normalizer = scale**2
    if fft:
        Ly = normalizer*scalespace(img, scale, order=(1, 0))
        Lx = normalizer*scalespace(img, scale, order=(0, 1))
    else:
        mode = 'reflect'
        Ly = normalizer*gaussian_filter(img, scale, order=(1, 0), mode=mode)
        Lx = normalizer*gaussian_filter(img, scale, order=(0, 1), mode=mode)
    if signed:
        go = np.arctan2(Ly, Lx)
    else:
        go = np.arctan(Ly/(Lx + 1e-10))
    go_m = np.sqrt(Lx**2+Ly**2)
    return go, go_m


def shape_index(img, scale, orientations=False, fft=False):
    '''Calculate the shape index at the given scale.'''
    normalizer = scale**2
    if fft:
        Lyy = normalizer*scalespace(img, scale, order=(2, 0))
        Lxy = normalizer*scalespace(img, scale, order=(1, 1))
        Lxx = normalizer*scalespace(img, scale, order=(0, 2))
    else:
        mode = 'reflect'
        Lyy = normalizer*gaussian_filter(img, scale, order=(2, 0), mode=mode)
        Lxy = normalizer*gaussian_filter(img, scale, order=(1, 1), mode=mode)
        Lxx = normalizer*gaussian_filter(img, scale, order=(0, 2), mode=mode)

    si = np.arctan((-Lxx-Lyy) / (np.sqrt((Lxx - Lyy)**2+4*Lxy**2)+1e-10))
    si_c = .5*np.sqrt(Lxx**2 + 2*Lxy**2 + Lyy**2)

    if orientations:
        t = Lxx + Lyy
        d = Lxx*Lyy - Lxy**2
        l1 = t/2.0 + np.sqrt(np.abs(t**2/4 - d))
        l2 = t/2.0 - np.sqrt(np.abs(t**2/4 - d))
        y = l1-Lyy
        x = Lxy
        si_o = np.arctan(y/(x+1e-10))
        si_om = l1-l2
        return si, si_c, si_o, si_om
    else:
        return si, si_c
