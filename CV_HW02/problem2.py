import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

#
# Task 1
#
def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    
    #
    # You code here/
    #

    obj = os.walk(path, topdown=False)
    filelist = []
    # os.walk() is like a tree, it has root, order(one low level from root),and files
    for root_dir, dir_names, file_names in obj:
        for file_name in file_names:
            filelist.append(os.path.join(root_dir, file_name))

    N = len(filelist)
    # get one random image and become h, w form this image
    # suppose each image has the same size
    img_rondom = plt.imread(filelist[6])
    h, w = img_rondom.shape
    M = h * w
    x = np.empty((N, M))
    for i in range(N):
        img = plt.imread(filelist[i])
        x[i, :] = img.reshape((1, M))

    return x, (h, w)

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (1, 2, 4)

#
# Task 3
#

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """
    X = X.T
    M, N = X.shape
    # calculate the mean x(mean vector of each dimension)
    # lecture 4 (folie 51)
    mean = np.sum(X, axis=1)/N
    X_hat = np.empty((M, N))
    for i in range(N):
        X_hat[:, i] = X[:, i] - mean
    u, s, v = np.linalg.svd(X_hat)
    u = u[:, 0:N]
    lmb = s*s/N

    return u, lmb
#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    M, N = u.shape
    s_sum = np.sum(s)
    D = 0
    percent = 0.0
    for i in range(N):
        D=D+1
        percent = percent+s[i]/s_sum
        if(percent >= p):break

    v = np.empty((M, D))
    for i in range(D):
        v[:, i] = u[:, i]
    
    return v

#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """
    # project image to u
    project = face_image.T @ u
    # project the projection back
    image_out = u @ project
    return image_out

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",
                # more return few error
                2: "The fewer principal components we use, the smaller is the re-projection error",
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                # e.g we can get the first eigenvalue from iris dataset only have 52%,not complete face
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return (1, 4, 5)


#
# Task 7
#
def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """
    Y_re = []
    # calculate Y features from projection
    feats = Y@u
    imgs = x@u
    # use ssd to calculate the distance
    diff = feats-imgs
    ssd = np.sum(np.power(diff, 2), axis=1)
    sort_index = np.argsort(ssd)
    for i in range(top_n):
        index = sort_index[i]
        Y_re.append(Y[index, :])

    Y_result = np.stack(Y_re, 0)
    return Y_result

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    # calculate the x1 and x2 project to u
    p1 = x1@u
    p2 = x2@u
    # interpolate into N
    M = np.linspace(p1, p2, N)
    Y = M@u.T

    return Y
