import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import griddata


######################
# Basic Lucas-Kanade #
######################

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape
    
    Ix = np.empty_like(im1)
    Iy = np.empty_like(im1)
    It = np.empty_like(im1)
    # the first derivative use prewitt filter
    x_filter = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]

    y_filter = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]

    Ix = convolve2d(im1, x_filter, boundary='symm', mode='same')
    Iy = convolve2d(im1, y_filter, boundary='symm', mode='same')
    It = im2-im1 # im2 is that im1 takes motion seconds later

    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It

def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    u = np.empty_like(Ix)
    v = np.empty_like(Iy)

    #if(aggregate=='gaussian'):
    h, w = Ix.shape
    kernel = np.ones((patch_size, patch_size))
    if (aggregate == 'gaussian' and isinstance(sigma, float)):
        kernel = (patch_size**2) * gaussian_kernel(patch_size, sigma)

    tensor_s1 = tensor_s2 = np.empty((h,w))
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy
    tensor_s1 = -Ix * It
    tensor_s2 = -Iy * It
    Ixx = convolve2d(Ixx, kernel, boundary='symm', mode='same')
    Iyy = convolve2d(Iyy, kernel, boundary='symm', mode='same')
    Ixy = convolve2d(Ixy, kernel, boundary='symm', mode='same')
    tensor_s1 = convolve2d(tensor_s1, kernel, boundary='symm', mode='same')
    tensor_s2 = convolve2d(tensor_s2, kernel, boundary='symm', mode='same')
    for i in range(h):
        for j in range(w):
            A = np.empty((2, 2))
            b = np.empty((2, 1))
            A[0,0] = Ixx[i,j]
            A[0,1] = Ixy[i,j]
            A[1,0] = Ixy[i,j]
            A[1,1] = Iyy[i,j]
            b[0] = tensor_s1[i,j]
            b[1] = tensor_s2[i,j]
            x = np.linalg.inv(A)@b
            u[i,j] = x[0]
            v[i,j] = x[1]

    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape
    
    im_warp = np.empty_like(im)

    h, w = im.shape
    im_points = np.empty((h*w, 2)) # 2d
    im_grids = np.empty((h*w, 2)) # 1d
    im_1d = im.reshape((h*w, 1))
    count = 0
    for i in range(h):# row
        for j in range(w):# line
            im_points[count, 0] = i # 0th is row
            im_points[count, 1] = j # 1th is line
            im_grids[count, 0] = i+v[i, j] # 0th is the change on row
            im_grids[count, 1] = j+u[i, j] # 1th is the change on line
            count += 1

    im_warp = griddata(im_points, im_1d, im_grids, method='nearest')
    im_warp=im_warp.reshape((h, w))

    assert im_warp.shape == im.shape
    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = 0.0

    h, w = im1.shape
    for i in range(h):
        for j in range(w):
            d += (im1[i, j]-im2[i, j])**2 # use ssd

    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#
def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: deviation of the Guassian

    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()

def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """

    h, w = x.shape
    gauss_x = convolve2d(x, gaussian_kernel(fsize, sigma), boundary='symm', mode='same')
    image = np.empty((int(h/2), int(w/2)))
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            image[int(i/2), int(j/2)] = gauss_x[i, j]

    return image

def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of gaussian downsampled images, it shoud be 3 * H * W
    '''
    GP = []
    image = np.copy(img)
    GP.append(image)
    while True:
        nlevels=nlevels-1
        if(nlevels>0):
            #image = convolve2d(image, gaussian_kernel(fsize, sigma), boundary='symm', mode='same')
            image = downsample_x2(image,fsize,sigma)
            GP.append(image)
        else:
            break
    GP.reverse()

    return GP


###############################
# Coarse-to-fine Lucas-Kanade #
###############################

# upscale when do iterations
def upscale(image):
    h,w = image.shape
    scale = np.empty((h*2, w*2))
    for i in range(h):
        for j in range(w):
            scale[2*i][2*j] = image[i][j]
            scale[2*i][2*j+1] = image[i][j]
            scale[2*i+1][2*j] = image[i][j]
            scale[2*i+1][2*j+1] = image[i][j]
    return scale

def coarse_to_fine(im1, im2, pyr1, pyr2, n_iter):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape

    u = np.zeros_like(im1)
    v = np.zeros_like(im1)
    image1 = np.copy(im1)
    image2 = np.copy(im2)
    image1_warp = np.copy(im1)
    len1 = len(pyr1)
    for i in range(n_iter):
        p1 = pyr1[0]
        p2 = pyr2[0]
        u_ = 0
        v_ = 0
        for n in range(len1):
            # compute derivatives
            Ix, Iy, It = compute_derivatives(p1, p2)
            # compute warp
            t_u, t_v = compute_motion(Ix, Iy, It)
            u_ = t_u + u_
            v_ = t_v + v_
            if n < (len1 - 1):
                u_ = upscale(u_)
                v_ = upscale(v_)
                p1 = pyr1[n + 1]
                p1 = warp(p1, u_, v_)
                p2 = pyr2[n + 1]
        u = u_ + u
        v = v_ + v
        image1 = warp(image1_warp, u, v)
        cost = compute_cost(image1, image2)
        if cost < 1e-6:
            break

    assert u.shape == im1.shape and \
           v.shape == im1.shape
    return u, v



###############################
#   Multiple-choice question  #
###############################
def task9_answer():
    """
    Which statements about optical flow estimation are true?
    Provide the corresponding indices in a tuple.

    1. For rectified image pairs, we can estimate optical flow 
       using disparity computation methods.
    2. Lucas-Kanade method allows to solve for large motions in a principled way
       (i.e. without compromise) by simply using a larger patch size.
    3. Coarse-to-fine Lucas-Kanade is robust (i.e. negligible difference in the 
       cost function) to the patch size, in contrast to the single-scale approach.
    4. Lucas-Kanade method implicitly incorporates smoothness constraints, i.e.
       that the flow vector of neighbouring pixels should be similar.
    5. Lucas-Kanade is not robust to brightness changes.

    """

    return (1, 4, 5)