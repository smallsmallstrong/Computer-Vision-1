import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math

def gaussian(sigma):
    """Computes (3, 1) array corresponding to a Gaussian filter.
    Normalisation is not required.

    Args:
        sigma: standard deviation used in the exponential

    Returns:
        gauss: numpy (3, 1) array of type float

    """
    gauss = np.empty((3, 1))
    #
    # You code goes here
    #
    gauss[0,0] = np.exp(-1/2/sigma/sigma)
    gauss[1,0] = 1
    gauss[2,0] = gauss[0,0]
    #print(gauss)
    return gauss

def diff():
    """Returns the derivative part corresponding to the central differences.
    The effect of this operator in x direction on function f should be:

            diff(x, y) = f(x + 1, y) - f(x - 1, y) 

    Returns:
        diff: (1, 3) array (float)
    """

    #
    # You code goes here
    #
    diff = np.empty((1, 3))
    diff[0,0] = -1
    diff[0,1] = 0
    diff[0,2] = 1
    return diff

def create_sobel():
    """Creates Sobel operator from two [3, 1] filters
    implemented in gaussian() and diff()

    Returns:
        sx: Sobel operator in x-direction
        sy: Sobel operator in y-direction
        sigma: Value of the sigma used to call gaussian()
        z: scaler of the operator
    """

    sigma = -9999
    z = -9999
    
    #
    # You code goes here
    # 
    sx = np.zeros((3, 3))
    sy = np.zeros((3, 3))
    
    sigma = sigmaold = 0.5
    while True:
        knew = gaussian(sigma)
        dnew = knew[1,0]/knew[0,0]
        kold = gaussian(sigmaold)
        dold = kold[1,0]/kold[0,0]
        if(abs(dnew-2)<0.01):
            break
        else:
            if dold>2 and dnew>2:
                sigmaold = sigma
                sigma = sigma+0.001
            elif dold<2 and dnew<2:
                sigmaold = sigma
                sigma = sigma-0.001
            else:
                sigmaoldcopy = sigmaold
                sigmaold = sigma
                sigma = (sigma+sigmaoldcopy)/2


    gaussian_kernel = np.around(gaussian(sigma),1)
    k = 2/gaussian_kernel[1,0]
    sx = np.outer(k*gaussian_kernel,diff())
    sy = sx.transpose()
    # do not change this
    return sx, sy, sigma, z

def apply_sobel(im, sx, sy):
    """Applies Sobel filters to a greyscale image im and returns
    L2-norm.

    Args:
        im: (H, W) image (greyscale)
        sx, sy: Sobel operators in x- and y-direction

    Returns:
        norm: L2-norm of the filtered result in x- and y-directions
    """

    im_norm = im.copy()

    #
    # Your code goes here
    #
    Gx = signal.convolve2d(im_norm,sx,mode='same')
    Gy = signal.convolve2d(im_norm,sy,mode='same')
    im_norm = np.sqrt(np.power(Gx,2)+np.power(Gy,2))
    return im_norm


def sobel_alpha(kx, ky, alpha):
    """Creates a steerable filter for give kx and ky filters and angle alpha.
    The effect the created filter should be equivalent to 
        cos(alpha) I*kx + sin(alpha) I*ky, where * stands for convolution.

    Args:
        kx, ky: (3x3) filters
        alpha: steering angle

    Returns:
        ka: resulting kernel
    """
     
    #
    # You code goes here
    #
    ka = kx*math.cos(alpha)+ky*math.sin(alpha)
    return ka


"""
This is a multiple-choice question
"""

class EdgeDetection(object):

    # choice of the method
    METHODS = {
                1: "hysteresis",
                2: "non-maximum suppression"
    }

    # choice of the explanation
    # by "magnitude" we mean the magnitude of the spatial gradient
    # by "maxima" we mean the maxima of the spatial gradient
    EFFECT = {
                1: "it sharpens the edges by retaining only the local maxima",
                2: "it weakens edges with high magnitude if connected to edges with low magnitude",
                3: "it recovers edges with low magnitude if connected to edges with high magnitude",
                4: "it makes the edges thicker with Gaussian smoothing",
                5: "it aligns the edges with a dominant orientation"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns tuples of two items: the first item
        is the method you will use and the second item is the explanation
        of its effect on the image. For example,
                ((2, 1), (1, 1))
        means "hysteresis sharpens the edges by retaining only the local maxima",
        and "non-maximum suppression sharpens the edges by retaining only the local maxima"
        
        Any wrong answer will cancel the correct answer.
        """

        return ((1,3),(2, 1))
