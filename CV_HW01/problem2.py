import numpy as np
import scipy.signal as signal

def rgb2bayer(image):
    """Convert image to bayer pattern:
    [B G]
    [G R]

    Args:
        image: Input image as (H,W,3) numpy array

    Returns:
        bayer: numpy array (H,W,3) of the same type as image
        where each color channel retains only the respective 
        values given by the bayer pattern
    """
    assert image.ndim == 3 and image.shape[-1] == 3

    # otherwise, the function is in-place
    bayer = image.copy()


    #red channel
    bayer[::2, :, 0] = 0
    bayer[:, ::2, 0] = 0
    #green channel
    bayer[:, :, 1][1::2, 1::2] = 0
    bayer[:, :, 1][::2, ::2] = 0
    #blue channel
    bayer[1::2, :, 2] = 0
    bayer[:, 1::2, 2] = 0

    assert bayer.ndim == 3 and bayer.shape[-1] == 3
    return bayer

def nearest_up_x2(x):
    """Upsamples a 2D-array by a factor of 2 using nearest-neighbor interpolation.

    Args:
        x: 2D numpy array (H, W)

    Returns:
        y: 2D numpy array if size (2*H, 2*W)
    """
    assert x.ndim == 2
    h, w = x.shape
    y = np.empty((2*h, 2*w))

    # for i in range(h-1):
    #     for j in range(w-1):
    #         y[i*2][j*2] = x[i][j]
    #         y[i*2+1][j*2] = x[i][j]
    #         y[i*2][j*2+1] = x[i][j]
    #         y[i*2+1][j*2+1] = x[i][j]

    y1 = np.repeat(x, 2, axis=0)
    y = np.repeat(y1, 2, axis=1)

    assert y.ndim == 2 and \
            y.shape[0] == 2*x.shape[0] and \
            y.shape[1] == 2*x.shape[1]
    return y

def bayer2rgb(bayer):
    """Interpolates missing values in the bayer pattern.
    Note, green uses bilinear interpolation; red and blue nearest-neighbour.

    Args:
        bayer: 2D array (H,W,C) of the bayer pattern
    
    Returns:
        image: 2D array (H,W,C) with missing values interpolated
        green_K: 2D array (3, 3) of the interpolation kernel used for green channel
        redblue_K: 2D array (3, 3) using for interpolating red and blue channels
    """
    assert bayer.ndim == 3 and bayer.shape[-1] == 3



    image = bayer.copy()
    rb_k = np.empty((3, 3))
    g_k = np.empty((3, 3))

    # bilinear interpolation
    #g_k = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])*1/4
    g_k = np.array([[0.5, 0, 0.5], [0, 0, 0], [0.5, 0, 0.5]])
    # nearest-neighbour interpolation
    rb_k = np.array([[2, 0, 2], [0, 0, 0], [2, 0, 2]])*1/2

    image[:, :, 0] = signal.convolve2d(image[:, :, 0], rb_k,
                                       mode='same', boundary='fill', fillvalue=0)
    image[:, :, 1] = signal.convolve2d(image[:, :, 1], g_k,
                                       mode='same', boundary='fill', fillvalue=0)
    image[:, :, 2] = signal.convolve2d(image[:, :, 2], rb_k,
                                       mode='same', boundary='fill', fillvalue=0)
    # res = signal.convolve2d(image[:, :, 0], rb_k, mode='same', boundary='fill', fillvalue=0)
    # print(res.shape)

    assert image.ndim == 3 and image.shape[-1] == 3 and \
                g_k.shape == (3, 3) and rb_k.shape == (3, 3)
    return image, g_k, rb_k

def scale_and_crop_x2(bayer):
    """Upscamples a 2D bayer pattern by factor 2 and takes the central crop.

    Args:
        bayer: 2D array (H, W) containing bayer pattern

    Returns:
        image_zoom: 2D array (H, W) corresponding to x2 zoomed and interpolated 
        one-channel image
    """
    assert bayer.ndim == 2
    #h,w = bayer.shape
    y = nearest_up_x2(bayer)
    y_h, y_w = y.shape
    bayer = y[int(y_h*0.25):int(y_h*0.75), int(y_w*0.25):int(y_w*0.75)]

    cropped = bayer.copy()

    assert cropped.ndim == 2
    return cropped
