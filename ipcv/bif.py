import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .scalespace import scalespace
from .misc import normalize


def bif_max(bif_r):
    return np.argmax(bif_r, axis=2)


def bif_response(img, scale, eps=0.0, fft=False):
    mode = 'reflect'
    if eps > 0.0:
        if fft:
            L = scalespace(img, scale, order=(0, 0))
        else:
            L = gaussian_filter(img, scale, order=(0, 0), mode=mode)
        bif_r = np.empty(img.shape + (7,))
        bif_r[..., 6] = eps*L
    else:
        bif_r = np.empty(img.shape + (6,))

    if fft:
        Ly = scale*scalespace(img, scale, order=(1, 0))
        Lx = scale*scalespace(img, scale, order=(0, 1))
        Lyy = scale**2*scalespace(img, scale, order=(2, 0))
        Lxy = scale**2*scalespace(img, scale, order=(1, 1))
        Lxx = scale**2*scalespace(img, scale, order=(0, 2))
    else:
        Ly = scale*gaussian_filter(img, scale, order=(1, 0), mode=mode)
        Lx = scale*gaussian_filter(img, scale, order=(0, 1), mode=mode)
        Lyy = scale**2*gaussian_filter(img, scale, order=(2, 0), mode=mode)
        Lxy = scale**2*gaussian_filter(img, scale, order=(1, 1), mode=mode)
        Lxx = scale**2*gaussian_filter(img, scale, order=(0, 2), mode=mode)

    lambd = Lyy+Lxx
    gamma = np.sqrt((Lyy-Lxx)**2 + 4*Lxy**2)

    bif_r[..., 0] = 2*np.sqrt(Ly**2+Lx**2)
    bif_r[..., 1] = lambd
    bif_r[..., 2] = -lambd
    bif_r[..., 3] = 2**(-.5)*(gamma+lambd)
    bif_r[..., 4] = 2**(-.5)*(gamma-lambd)
    bif_r[..., 5] = gamma

    return bif_r


def bif_colors(bif_r):
    img = np.zeros(bif_r.shape[0:2] + (3,))
    if bif_r.ndim == 2:
        max_idx = bif_r
    else:
        max_idx = bif_max(bif_r)
    img[max_idx == 0] = [.5, .5, .5]
    img[max_idx == 1] = [0, 0, 0]
    img[max_idx == 2] = [1, 1, 1]
    img[max_idx == 3] = [0, 0, 1]
    img[max_idx == 4] = [1, 1, 0]
    img[max_idx == 5] = [0, 1, 0]
    img[max_idx == 6] = [1, 0, 0]
    return img


def bif_hist(img, n_scales=4, scale_min=1.0, scale_ratio=2.0, eps=0.0,
             norm='l1'):
    if eps > 0.0:
        nresponses = 7
    else:
        nresponses = 6
    # Classify image structure at all scales
    scales = [scale_min*scale_ratio**n for n in range(n_scales)]
    bif_maxes = [bif_max(bif_response(img, s, eps)) for s in scales]
    bif_maxes = np.concatenate([b[..., np.newaxis] for b in bif_maxes], axis=2)
    # Calculate histogram indices for all pixels
    offsets = nresponses**np.arange(n_scales)
    hist_idx = np.sum(bif_maxes * offsets[np.newaxis, np.newaxis, :], axis=2)
    # Build histogram
    hist_dims = nresponses**n_scales
    hist = np.bincount(np.ravel(hist_idx), minlength=hist_dims)
    hist = normalize(hist.astype(float), norm)
    return hist
