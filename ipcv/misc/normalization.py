import numpy as np


def normalize(x, method):
    '''Normalize vector

    Normalize vector according to the specified method.

    Args:
        x: vector to be normalized.
        method: One of the following normalization methods:
            ['l1', 'l1_root', 'l2', 'none'].

    Returns:
        Normalized vector with the same shape as x.
    '''
    if method == 'l1_root':
        x = x/np.sum(x)
        x = np.sqrt(x)
        x = x/np.sum(x)
    elif method == 'l1':
        x = x/np.sum(x)
    elif method == 'l2':
        x = x/np.sqrt(np.sum(x**2))
    elif method == 'none':
        pass
    else:
        raise ValueError('Invalid normalization method.')
    return x
