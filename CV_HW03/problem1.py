import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import convolve2d

#
# Hint: you can make use of this function
# to create Gaussian kernels for different sigmas
#
def gaussian_kernel(fsize=7, sigma=1):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of guassian kernel

    Returns:
        Gaussian kernel
    '''
    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return G / (2 * math.pi * sigma**2)

def load_image(path):
    '''
    The input image is a) loaded, b) converted to greyscale, and
     c) converted to numpy array [-1, 1].

    Args:
        path: the name of the inpit image
    Returns:
        img: numpy array containing image in greyscale
    '''
    image = plt.imread(path)
    height, width, channels = image.shape
    gray_img = np.empty((height, width))
    sum = 255.0
    for i in range(height):
        for j in range(width):
            gray_img[i, j] = 2 * (0.2989 * image[i, j, 0] + 0.5870 * image[i, j, 1] + 0.1140 * image[i, j, 2]) / sum - 1
    return gray_img

def smoothed_laplacian(image, sigmas, lap_kernel):
    """
    Then laplacian operator is applied to the image and
     smoothed by gaussian kernels for each sigma in the list of sigmas.


    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    """
    n = len(sigmas)
    res = np.empty((n, *image.shape))
    tempimg = convolve2d(image, lap_kernel, boundary='symm', mode='same')
    for i in range(n):
        g_kernel = gaussian_kernel(7, sigmas[i])
        g_kernel = g_kernel/g_kernel.sum()
        res[i] = convolve2d(tempimg,g_kernel, boundary='symm', mode='same')
    return res

def laplacian_of_gaussian(image, sigmas):
    """
    Then laplacian of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    """
    n = len(sigmas)
    res = np.empty((n, *image.shape))
    for i in range(n):
        res[i] = convolve2d(image, LoG_kernel(9, sigmas[i]), boundary='symm', mode='same')
    return res

def difference_of_gaussian(image, sigmas):
    '''
    Then difference of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''
    n = len(sigmas)
    res = np.empty((n, *image.shape))
    for i in range(n):
        res[i] = convolve2d(image, DoG(sigmas[i]), boundary='symm', mode='same')
    return res

def LoG_kernel(fsize=9, sigma=1):
    '''
    Define a LoG kernel.
    Tip: First calculate the second derivative of a gaussian and then discretize it.
    Args:
        fsize: kernel size
        sigma: sigma of guassian kernel

    Returns:
        LoG kernel
    '''
    log_kernel = np.empty((fsize, fsize))
    for i in range(fsize):
        for j in range(fsize):
            x = i-(fsize-1)/2
            y = j-(fsize-1)/2
            log_kernel[i, j] = (x*x+y*y-2*sigma*sigma)/math.pow(sigma, 4)*np.exp(-(x*x+y*y)/(2*sigma*sigma))

    return log_kernel


def blob_detector(response):
    """
    Find unique extrema points (maximum or minimum) in the response using 9x9 spatial neighborhood
    and across the complete scale dimension.
    Tip: Ignore the boundary windows to avoid the need of padding for simplicity.
    Tip 2: unique here means skipping an extrema point if there is another point in the local window
            with the same value
    Args:
        response: 3 dimensional response from LoG operator in scale space.

    Returns:
        list of 3-tuples (scale_index, row, column) containing the detected points.
    """

    l, h, w = response.shape
    res_list = []
    min_percent = np.percentile(response, 0.1)
    max_percent = np.percentile(response, 99.9)
    for i in range(h - 8):
        for j in range(w - 8):

            temparray = response[0:l, i:i + 9, j:j + 9]
            maximun = np.amax(temparray)
            if (maximun >= max_percent and len(np.where(temparray == maximun)[0]) == 1):
                scale_index, target_i, target_j = np.unravel_index(temparray.argmax(), temparray.shape)
                if (target_i == 4 and target_j == 4):
                    target_j = target_j + j
                    target_i = target_i + i
                    res_list.append((scale_index, target_i, target_j))

            minimun = np.amin(temparray)
            if (minimun <= min_percent and len(np.where(temparray == minimun)[0]) == 1):
                scale_index, target_i, target_j = np.unravel_index(temparray.argmin(), temparray.shape)
                if (target_i == 4 and target_j == 4):
                    target_j = target_j + j
                    target_i = target_i + i
                    res_list.append((scale_index, target_i, target_j))

    return res_list


def DoG(sigma):
    '''
    Define a DoG kernel. Please, use 9x9 kernels.
    Tip: First calculate the two gaussian kernels and return their difference. This is an approximation for LoG.

    Args:
        sigma: sigma of guassian kernel

    Returns:
        DoG kernel
    '''


    sigma1 = math.sqrt(2)*sigma
    sigma2 = sigma/math.sqrt(2)
    g_kernel_1 = gaussian_kernel(9, sigma1)
    g_kernel_2 = gaussian_kernel(9, sigma2)
    dog_kernel = (g_kernel_1-g_kernel_2)
    return dog_kernel


def laplacian_kernel():
    '''
    Define a 3x3 laplacian kernel.
    Tip1: I_xx + I_yy
    Tip2: There are two possible correct answers.
    Args:
        none

    Returns:
        laplacian kernel
    '''
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]])
    return kernel


class Method(object):

    # select one or more options
    REASONING = {
        1: 'it is always more computationally efficient',
        2: 'it is always more precise.',
        3: 'it always has fewer singular points',
        4: 'it can be implemented with convolution',
        5: 'All of the above are incorrect.'
    }

    def answer(self):
        '''Provide answer in the return value.
        This function returns a tuple containing indices of the correct answer.
        '''

        return (1,2)
