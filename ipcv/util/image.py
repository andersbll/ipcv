import os
import numpy as np
import scipy as sp


def stretch_intensity(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img)+1e-12
    return img


def imsave(path, img, stretch=True):
    if stretch:
        img = (255*stretch_intensity(img)).astype(np.uint8)
    dirpath = os.path.dirname(path)
    if len(dirpath) > 0 and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    sp.misc.imsave(path, img)


def tile(imgs, aspect_ratio=1.0, tile_shape=None):
    ''' Tile images in a grid.

    If tile_shape is provided only as many images as specified in tile_shape
    will be included in the output.
    '''

    # Prepare images
    imgs = np.array(imgs)
    n_imgs = imgs.shape[0]
    img_shape = imgs.shape[1:]
    assert len(img_shape) == 2 or len(img_shape) == 3
    if len(img_shape) == 2:
        # Add color dimension to greyscale images. This allows the rest of the
        # code to assume the image has a color dimension.
        imgs = np.reshape(imgs, imgs.shape + (1,))
        img_shape = img_shape+(1,)

    # Calculate grid shape
    img_shape = np.array(img_shape)
    if tile_shape is None:
        img_aspect_ratio = img_shape[0] / float(img_shape[1])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs) * aspect_ratio))
        tile_width = int(np.ceil(np.sqrt(n_imgs) * 1/aspect_ratio))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Calculate tile image shape
    spacing = 1
    tile_img_shape = img_shape.copy()
    tile_img_shape[:2] = (img_shape[:2] + spacing) * grid_shape[:2] - spacing

    # Assemble tile image
    tile_img = np.ones(tile_img_shape)*np.min(imgs)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i*grid_shape[0]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + spacing) * i
            xoff = (img_shape[1] + spacing) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], :] = img

    # Squeeze color channel away if image is greyscale
    tile_img = np.squeeze(tile_img)
    return tile_img


def patch(img, pt, patch_size, padding='none'):
    radius = patch_size//2
    y_min = max(0, pt[0]-radius)
    x_min = max(0, pt[1]-radius)
    if patch_size % 2 == 0:
        y_max = min(img.shape[0]-1, pt[0]+radius)
        x_max = min(img.shape[1]-1, pt[1]+radius)
    else:
        y_max = min(img.shape[0]-1, pt[0]+radius+1)
        x_max = min(img.shape[1]-1, pt[1]+radius+1)

    if padding == 'none':
        patch = img[y_min:y_max, x_min:x_max, ...]
    else:
        y_offset = max(0, -(pt[0]-radius))
        x_offset = max(0, -(pt[1]-radius))
        height = y_max-y_min
        width = x_max-x_min
        if padding == 'zero':
            patch = np.zeros((patch_size, patch_size))
            patch[y_offset:y_offset+height, x_offset:x_offset+width, ...] = \
                img[y_min:y_max, x_min:x_max, ...]
        elif padding == 'mirror':
            raise NotImplementedError
        else:
            raise ValueError('Invalid padding method')
    return patch


def extract_patches(imgs, n_patches, patch_size, random_seed=None,
                    filter_fun=None):
    patches = np.empty((n_patches, patch_size, patch_size)+(imgs[0].shape[2:]))
    i = 0
    padding = patch_size//2+1
    if random_seed is not None:
        np.random.seed(random_seed)
    while i < n_patches:
        img_idx = np.random.randint(0, len(imgs))
        img = imgs[img_idx]
        y = np.random.randint(padding, img.shape[0]-padding)
        x = np.random.randint(padding, img.shape[1]-padding)
        patches[i, ...] = patch(img, (y, x), patch_size)
        if filter_fun is not None:
            if not filter_fun(patches[i, ...]):
                continue
        i += 1
    return patches


def pad(img, padding, fill_value=0):
    if type(padding) is tuple:
        if len(padding) == 2:
            top = bottom = padding[0]
            right = left = padding[1]
        elif len(padding) == 4:
            top, right, bottom, left = padding
        else:
            raise ValueError('padding must be either an integer, a two-tuple' +
                             ', or a four-tuple')
    else:
        top = right = bottom = left = padding
    height, width = img.shape[:2]
    img_padded = np.ones((height+top+bottom, width+right+left) + img.shape[2:])
    img_padded *= fill_value
    img_padded[top:top+height, left:left+width, ...] = img
    return img_padded
