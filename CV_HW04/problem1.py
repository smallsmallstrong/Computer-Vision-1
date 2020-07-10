import numpy as np
from scipy.linalg import null_space

def transform(pts):
    """Point conditioning: scale and shift points into [-1, 1] range
    as suggested in the lecture.
    
    Args:
        pts: [Nx2] numpy array of pixel point coordinates
    
    Returns:
        T: [3x3] numpy array, such that Tx normalises 
            2D points x (in homogeneous form).
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2

    #
    # Your code goes here
    #
    T = np.zeros((3, 3))

    pointsnorm = np.linalg.norm(pts, axis=1)
    s = np.amax(pointsnorm)/2
    nr,nc = pts.shape
    t = np.sum(pts,axis=0)/nr
    T[0,0] = 1/s
    T[1,1] = 1/s
    T[0,2] = -t[0]/s
    T[1,2] = -t[1]/s
    T[2,2] = 1
    assert T.shape == (3, 3)
    return T


def transform_pts(pts, T):
    """Applies transformation T on 2D points.
    
    Args:
        pts: (Nx2) 2D point coordinates
        T: 3x3 transformation matrix
    
    Returns:
        pts_out: (Nx3) transformed points in homogeneous form
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert T.shape == (3, 3)

    #
    # Your code goes here
    #
    #pts_h = np.empty((pts.shape[0], 3))
    nr,nc = pts.shape
    extercol = np.ones((nr, 1))
    pts_h = np.hstack((pts, extercol))#N*3
    pts_h = pts_h@T.T#N*3 dot 3*3 = N*3
    assert pts_h.shape == (pts.shape[0], 3)
    return pts_h

def create_A(pts1, pts2): #pts1:x,pts2:x'
    """Create matrix A such that our problem will be Ax = 0,
    where x is a vectorised representation of the 
    fundamental matrix.
        
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points 
    
    Returns:
        A: numpy array
    """
    assert pts1.shape == pts2.shape

    #
    # Your code goes here
    #
    nr,nc = pts1.shape
    A = np.empty((nr,9))
    for i in range(nr):
        A[i, 0] = pts1[i,0]*pts2[i,0]
        A[i, 1] = pts1[i,1]*pts2[i,0]
        A[i, 2] = pts2[i,0]
        A[i, 3] = pts1[i,0]*pts2[i,1]
        A[i, 4] = pts1[i,1]*pts2[i,1]
        A[i, 5] = pts2[i,1]
        A[i, 6] = pts1[i,0]
        A[i, 7] = pts1[i,1]
        A[i, 8] = 1
    return A

def enforce_rank2(F):
    """Enforce rank 2 of 3x3 matrix
    
    Args:
        F: 3x3 matrix
    
    Returns:
        F_out: 3x3 matrix with rank 2
    """
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #

    u, s, v = np.linalg.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F_final = u @ s @ v

    assert F_final.shape == (3, 3)
    return F_final

def compute_F(A):
    """Computing the fundamental matrix from F
    by solving homogeneous least-squares problem
    Ax = 0, subject to ||x|| = 1
    
    Args:
        A: matrix A
    
    Returns:
        f: 3x3 matrix subject to rank-2 contraint
    """
    
    #
    # Your code goes here
    #
    #F_final = np.empty((3, 3))
    u, s, v = np.linalg.svd(A)
    F = v[8,:].reshape((3, 3))
    F_final = enforce_rank2(F)
    assert F_final.shape == (3, 3)
    return F_final

def compute_residual(F, x1, x2):#x1:x,x2:x'
    """Computes the residual g as defined in the assignment sheet.
    
    Args:
        F: fundamental matrix
        x1,x2: point correspondences
    
    Returns:
        float
    """

    #
    # Your code goes here
    #
    nr,nc = x1.shape # = N*2
    extercol = np.ones((nr, 1))
    x1_temp = np.hstack((x1, extercol))# N*3
    x2_temp = np.hstack((x2, extercol))
    sum_temp = 0
    for i in range(nr):
        temp_res = x2_temp[i,:]@F@x1_temp[i,:].T #1*3 dot 3*3 dot 3*1 = 1*1
        sum_temp = sum_temp + abs(temp_res)

    return sum_temp/nr

def denorm(F, T1, T2):#T1:T , T2:T'
    """Denormalising matrix F using 
    transformations T1 and T2 which we used
    to normalise point coordinates x1 and x2,
    respectively.
    
    Returns:
        3x3 denormalised matrix F
    """

    #
    # Your code goes here
    #
    F = T2.T@F@T1
    return F.copy()

def estimate_F(x1, x2, t_func): #x1:x,x2:x'
    """Estimating fundamental matrix from pixel point
    coordinates x1 and x2 and normalisation specified 
    by function t_func (t_func returns 3x3 transformation 
    matrix).
    
    Args:
        x1, x2: 2D pixel coordinates of matching pairs
        t_func: normalising function (for example, transform)
    
    Returns:
        F: fundamental matrix
        res: residual g defined in the assignment
    """
    
    assert x1.shape[0] == x2.shape[0]

    #
    # Your code goes here
    #
    T1 = t_func(x1)#x -> T
    T2 = t_func(x2)#x' -> T'

    x1_normalized = transform_pts(x1, T1)
    x2_normalized = transform_pts(x2, T2)

    A = create_A(x1_normalized,x2_normalized)
    F = compute_F(A)
    F = denorm(F,T1,T2)
    res = compute_residual(F, x1, x2)#x1:x,x2:x'

    return F, res


def line_y(xs, F, pts):
    """Compute corresponding y coordinates for 
    each x following the epipolar line for
    every point in pts.
    
    Args:
        xs: N-array of x coordinates
        F: fundamental matrix
        pts: (Mx3) array specifying pixel corrdinates
             in homogeneous form.
    
    Returns:
        MxN array containing y coordinates of epipolar lines.
    """

    N, M = xs.shape[0], pts.shape[0]
    assert F.shape == (3, 3)
    #
    # Your code goes here
    #
    ys = np.empty((M, N))
    for i in range(M):
        k = pts[i,:]@F #1*3 dot 3*3 = 1*3
        ys[i,:] = -xs*(k[0]/k[1])-k[2]/k[1]

    assert ys.shape == (M, N)
    return ys


#
# Bonus tasks
#

import math


def transform_v2(pts):
    """Point conditioning: scale and shift points into [-1, 1] range.
    
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points
    
    Returns:
        T: numpy array, such that Tx conditions 2D (homogeneous) points x.
    
    """
    
    #
    # Your code goes here
    #
    T = np.zeros((3, 3))
    nr, nc = pts.shape
    t = np.sum(pts, axis=0) / nr
    pointsnorm = np.linalg.norm(pts, axis=1)
    d = np.sum(pointsnorm) / nr
    s = math.sqrt(2)/d
    T[0,0] = s
    T[1,1] = s
    T[2,2] = 1
    T[0,2] = -t[0]*s
    T[1,2] = -t[1]*s

    return T


"""Multiple-choice question"""
class MultiChoice(object):

    """ Which statements about fundamental matrix F estimation are true?

    1. We need at least 7 point correspondences to estimate matrix F.
    2. We need at least 8 point correspondences to estimate matrix F.
    3. More point correspondences will not improve accuracy of F as long as 
    the minimum number of points correspondences are provided.
    4. Fundamental matrix contains information about intrinsic camera parameters.
    5. One can recover the rotation and translation (up to scale) from the essential matrix 
    corresponding to the transform between the two views.
    6. The determinant of the fundamental matrix is always 1.
    7. Different normalisation schemes (e.g. transform, transform_v2) may have
    a significant effect on estimation of F. For example, epipoles can deviate.
    (Hint for 7): Try using corridor image pair.)

    Please, provide the indices of correct options in your answer.
    """
    #
    def answer(self):
        return [1,4,7]


def compute_epipole(F, eps=1e-8):
    """Compute epipole for matrix F,
    such that Fe = 0.
    
    Args:
        F: fundamental matrix
    
    Returns:
        e: 2D vector of the epipole
    """
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #
    e = null_space(F, eps)
    return e[0:2,0]

def intrinsics_K(f=1.05, h=480, w=640):
    """Return 3x3 camera matrix.
    
    Args:
        f: focal length (same for x and y)
        h, w: image height and width
    
    Returns:
        3x3 camera matrix
    """

    #
    # Your code goes here
    #
    #K = np.empty((3, 3))
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    a[0,0] = w
    a[1,1] = h
    a[2,2] = 1
    b[0,0] = f
    b[1,1] = f
    b[2,2] = 1
    b[0,2] = w/2
    b[1,2] = h/2
    K = a@b
    return K

def compute_E(F):
    """Compute essential matrix from the provided
    fundamental matrix using the camera matrix (make 
    use of intrinsics_K).

    Args:
        F: 3x3 fundamental matrix

    Returns:
        E: 3x3 essential matrix
    """

    #
    # Your code goes here
    #
    #E = np.empty((3, 3))
    K = intrinsics_K()
    E = K.T@F@K

    return E