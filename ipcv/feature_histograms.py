import numpy as np
from .scalespace import gradient_orientation, shape_index
from .misc import normalize, isophotes


def go_hist(img, n_scales=4, scale_min=1.0, scale_ratio=2.0, n_bins=8,
            tonal_scale=0.25, norm='l1', weights=None, signed=True,
            ori_offsets=None):
    '''Gradient orientation histograms

    Generate a multi-scale gradient orientation histogram for the supplied
    image.

    Args:
        img: Grayscale image as a (p,q) array.
        n_scales: Number of scales for extracting gradients.
        scale_min: The smallest scale at which gradients are extracted.
        scale_ratio: The ratio between two consecutive scales.
        n_bins: Number of histogram bins.
        tonal_scale: The smoothing scale in the orientation dimension.
        norm: Histogram normalization method.
        signed: Use signed (360 deg.) or unsigned (180 deg.) orientations.
        ori_offsets: A (p,q) array of pixel-wise offsets for the gradient
            orientations.
        weights: A list of (p,q) arrays with pixel-wise weights to adjust
            histogram contributions.

    Returns:
        A gradient orientation histogram vector of dimensionality n_scales
        * n_bins.
    '''
    hists = []
    scales = [float(scale_min)*scale_ratio**i for i in range(n_scales)]
    if signed:
        limits = (-np.pi, np.pi)
    else:
        limits = (-np.pi/2, np.pi/2)
    for s in scales:
        go, go_m = gradient_orientation(img, s, signed)
        if ori_offsets is not None:
            if signed:
                go = np.mod(go-ori_offsets, 2*np.pi)
            else:
                go = np.mod(go-ori_offsets, np.pi)-np.pi/2
        if weights is None:
            weights_ = [go_m]
        else:
            weights_ = [w*go_m for w in weights]
        for w in weights_:
            iso = isophotes(go, n_bins, limits, tonal_scale, 'von_mises')
            iso *= w[np.newaxis, ...]
            h = np.sum(iso, axis=(1, 2))
            hists.append(h)
    hists = np.hstack(hists)
    hists = normalize(hists, norm)
    return hists


def si_hist(img, n_scales=4, scale_min=1.0, scale_ratio=2.0, n_bins=8,
            tonal_scale=0.25, norm='l1', weights=None):
    '''Shape index histograms

    Generate a multi-scale shape index histogram for the supplied image.

    Args:
        img: Grayscale image as a (p,q) array.
        n_scales: Number of scales for extracting shape indices.
        scale_min: The smallest scale at which the shape indices are extracted.
        scale_ratio: The ratio between two consecutive scales.
        n_bins: Number of histogram bins.
        tonal_scale: The smoothing scale in the shape index dimension.
        norm: Histogram normalization method.
        weights: A list of (p,q) arrays with pixel-wise weights to adjust
            histogram contributions.

    Returns:
        A shape index histogram vector of dimensionality n_scales * n_bins.
    '''
    hists = []
    scales = [float(scale_min)*scale_ratio**i for i in range(n_scales)]
    for s in scales:
        si, si_c = shape_index(img, s)
        si_iso = isophotes(si, n_bins, (-np.pi/2, np.pi/2), tonal_scale)
        if weights is None:
            weights_ = [si_c]
        else:
            weights_ = [w*si_c for w in weights]
        for w in weights_:
            h = np.sum(si_iso*w[np.newaxis, ...], axis=(1, 2))
            hists.append(h)
    hists = np.hstack(hists)
    hists = normalize(hists, norm)
    return hists


def osi_hist(img, n_scales=4, scale_min=1.0, scale_ratio=2.0, n_bins=8,
             tonal_scale=0.25, ori_n_bins=8, ori_tonal_scale=0.25, norm='l1',
             weights=None, ori_offsets=None, joint_hist=False):
    '''Oriented shape index histograms

    Generate a multi-scale oriented shape index histogram for the supplied
    image.

    Args:
        img: Grayscale image as a (p,q) array.
        n_scales: Number of scales for extracting shape indices.
        scale_min: The smallest scale at which the shape indices are extracted.
        scale_ratio: The ratio between two consecutive scales.
        n_bins: Number of histogram bins.
        tonal_scale: The smoothing scale in the shape index dimension.
        norm: Histogram normalization method.
        weights: A list of (p,q) arrays with pixel-wise weights to adjust
            histogram contributions.

    Returns:
        A shape index histogram vector of dimensionality n_scales * n_bins.
    '''
    hists = []
    hists_ori = []
    scales = [float(scale_min)*scale_ratio**i for i in range(n_scales)]
    for s in scales:
        si, si_c, si_o, si_om = shape_index(img, s, orientations=True)
        if ori_offsets is not None:
            si_o = np.mod(si_o+ori_offsets, np.pi)-np.pi/2
        iso_si = isophotes(si, n_bins, (-np.pi/2, np.pi/2), tonal_scale)
        iso_si_o = isophotes(si_o, ori_n_bins, (-np.pi/2, np.pi/2),
                             ori_tonal_scale, 'von_mises')
        if joint_hist:
            osih = (iso_si[np.newaxis, ...] * si_c
                    * iso_si_o[:, np.newaxis, ...] * si_om)
            if weights is None:
                h = np.sum(osih, axis=(2, 3))
                hists.append(np.ravel(h))
            else:
                for w in weights:
                    h = np.sum(osih*w, axis=(2, 3))
                    hists.append(np.ravel(h))
        else:
            iso_si *= si_c
            iso_si_o *= si_om
            if weights is None:
                h_si = np.sum(iso_si, axis=(1, 2))
                h_si_o = np.sum(iso_si_o, axis=(1, 2))
                hists.append(h_si)
                hists_ori.append(h_si_o)
            else:
                for w in weights:
                    h_si = np.sum(iso_si*w, axis=(1, 2))
                    h_si_o = np.sum(iso_si_o*w, axis=(1, 2))
                    hists.append(h_si)
                    hists_ori.append(h_si_o)
    hists = np.hstack(hists)
    hists = normalize(hists, norm)
    if joint_hist:
        return hists
    else:
        hists_ori = np.hstack(hists_ori)
        hists_ori = normalize(hists_ori, norm)
        return np.hstack([hists, hists_ori])
