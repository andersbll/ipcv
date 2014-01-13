import numpy as np
from .scalespace import gradient_orientation, shape_index
from .misc import normalize, isophotes


def go_hist(img, n_scales=4, scale_min=1.0, scale_ratio=2.0, n_bins=8,
            tonal_scale=0.4, norm='l1', weights=None, signed=True,
            ori_offsets=None):
    '''Gradient orientation histograms

    Compute a multi-scale gradient orientation histogram for the given image.

    Parameters
    ----------
    img: (h, w) array.
        Input image.
    n_scales: int
        Number of scales.
    scale_min: float
        The smallest scale at which gradients are extracted.
    scale_ratio: float
        The ratio between two consecutive scales.
    n_bins: int
        Number of histogram bins.
    tonal_scale: float
        The smoothing scale in the orientation dimension.
    norm: str
        Histogram normalization method.
    signed: bool
        Use signed (360 deg.) or unsigned (180 deg.) orientations.
    ori_offsets: (h, w) array
        Pixel-wise offsets for the gradient orientations.
    weights: A list of (h, w) arrays
        Pixel-wise spatial weights to adjust histogram contributions.

    Returns
    -------
    hists: (n_bins, n_scales) or (n_bins, n_scales, len(weights)) array
        Gradient orientation histograms for all combinations of scales and
        spatial weights.
    '''
    hists_shape = (n_bins, n_scales)
    if weights is not None:
        hists_shape += (len(weights), )
    hists = np.empty(hists_shape)
    scales = scale_min*scale_ratio**np.arange(n_scales)
    if signed:
        limits = (-np.pi, np.pi)
    else:
        limits = (-np.pi/2, np.pi/2)
    for s_idx, s in enumerate(scales):
        go, go_m = gradient_orientation(img, s, signed)
        if ori_offsets is not None:
            if signed:
                go = np.mod(go-ori_offsets, 2*np.pi)
            else:
                go = np.mod(go-ori_offsets, np.pi)-np.pi/2
        go_iso = isophotes(go, n_bins, limits, tonal_scale, 'von_mises') * go_m
        if weights is None:
            hists[:, s_idx] = np.sum(go_iso, axis=(1, 2))
        else:
            for w_idx, w in enumerate(weights):
                hists[:, s_idx, w_idx] = np.sum(go_iso*w, axis=(1, 2))
    hists = normalize(hists, norm)
    return hists


def si_hist(img, n_scales=4, scale_min=1.0, scale_ratio=2.0, n_bins=8,
            tonal_scale=0.25, norm='l1', weights=None):
    '''Shape index histograms

    Compute a multi-scale shape index histogram for the given image.

    Parameters
    ----------
    img: (h, w) array.
        Input image.
    n_scales: int
        Number of scales.
    scale_min: float
        The smallest scale at which gradients are extracted.
    scale_ratio: float
        The ratio between two consecutive scales.
    n_bins: int
        Number of histogram bins.
    tonal_scale: float
        The smoothing scale in the shape index dimension.
    norm: str
        Histogram normalization method.
    weights: A list of (h, w) arrays
        Pixel-wise spatial weights to adjust histogram contributions.

    Returns
    -------
    hists: (n_bins, n_scales) or (n_bins, n_scales, len(weights)) array
        Shape index histograms for all combinations of scales and spatial
        weights.
    '''
    hists_shape = (n_bins, n_scales)
    if weights is not None:
        hists_shape += (len(weights),)
    hists = np.empty(hists_shape)
    scales = scale_min*scale_ratio**np.arange(n_scales)
    for s_idx, s in enumerate(scales):
        si, si_c = shape_index(img, s)
        si_iso = isophotes(si, n_bins, (-np.pi/2, np.pi/2), tonal_scale)*si_c
        if weights is None:
            hists[:, s_idx] = np.sum(si_iso, axis=(1, 2))
        else:
            for w_idx, w in enumerate(weights):
                hists[:, s_idx, w_idx] = np.sum(si_iso*w, axis=(1, 2))
    hists = normalize(hists, norm)
    return hists


def osi_hist(img, n_scales=4, scale_min=1.0, scale_ratio=2.0, n_bins=8,
             tonal_scale=0.25, ori_n_bins=8, ori_tonal_scale=0.25, norm='l1',
             weights=None, ori_offsets=None, joint_hist=False):
    '''Oriented shape index histograms

    Compute a multi-scale oriented shape index histogram for the given image.

    Parameters
    ----------
    img: (h, w) array.
        Input image.
    n_scales: int
        Number of scales.
    scale_min: float
        The smallest scale at which gradients are extracted.
    scale_ratio: float
        The ratio between two consecutive scales.
    n_bins: int
        Number of histogram bins for the shape index.
    tonal_scale: float
        The smoothing scale in the shape index dimension.
    ori_n_bins: int
        Number of histogram bins for the shape index orientation.
    ori_tonal_scale: float
        The smoothing scale in the shape index orientation dimension.
    norm: str
        Histogram normalization method.
    weights: A list of (h, w) arrays
        Pixel-wise spatial weights to adjust histogram contributions.
    ori_offsets: (h, w) array
        Pixel-wise offsets for the shape index orientations.
    joint_hist: bool
        Choose between the joint histogram or two marginalized histograms.

    Returns
    -------
    hists: array or two-tuple of arrays
        If joint_hist: Joint oriented shape index histograms for all
        combinations of scales and spatial weights are returned as a (n_bins,
        ori_n_bins, n_scales) or (n_bins, ori_n_bins, n_scales, len(weights))
        array.
        If not joint_hist: Marginal oriented shape index histograms for all
        combinations of scales and spatial weights are returned as a two-tuple
        of arrays: ((n_bins, n_scales), (ori_n_bins, n_scales)) or ((n_bins,
        n_scales, len(weights)), (ori_n_bins, n_scales, len(weights)))
    '''
    if weights is None:
        hists_shape = (n_bins, ori_n_bins, n_scales)
    else:
        hists_shape = (n_bins, ori_n_bins, n_scales, len(weights))
    hists = np.empty(hists_shape)
    scales = scale_min*scale_ratio**np.arange(n_scales)
    for s_idx, s in enumerate(scales):
        si, si_c, si_o, si_om = shape_index(img, s, orientations=True)
        if ori_offsets is not None:
            si_o = np.mod(si_o+ori_offsets, np.pi)-np.pi/2
        # Smooth bin contributions (= soft isophote images)
        iso_si = isophotes(si, n_bins, (-np.pi/2, np.pi/2), tonal_scale)
        iso_si_o = isophotes(si_o, ori_n_bins, (-np.pi/2, np.pi/2),
                             ori_tonal_scale, 'von_mises')
        # Bin contributions for the joint histogram
        iso_j = (iso_si[np.newaxis, ...] * si_c
                 * iso_si_o[:, np.newaxis, ...] * si_om)
        # Summarize bin contributions in the joint histograms
        if weights is None:
            hists[:, :, s_idx] = np.sum(iso_j, axis=(2, 3))
        else:
            for w_idx, w in enumerate(weights):
                hists[:, :, s_idx, w_idx] = np.sum(iso_j*w, axis=(2, 3))

    if joint_hist:
        hists = normalize(hists, norm)
        return hists
    else:
        hists_si = np.sum(hists, axis=1)
        hists_ori = np.sum(hists, axis=0)
        hists_si = normalize(hists_si, norm)
        hists_ori = normalize(hists_ori, norm)
        return (hists_si, hists_ori)
