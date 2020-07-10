import numpy as np
import scipy.linalg as lg

def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy). 
    Image points are saved in 'image'
    Object points are saved in 'world'

    Returns:
        image: A Nx2 array of 2D points form image coordinate 
        world: A N*3 array of 3D points form world coordinate
    '''
    # read the file, and give file to variable
    file = np.load(path)
    image_pts = np.empty((75, 3))
    world_pts = np.empty((75, 4))
    image_pts, world_pts = file['image'], file['world']

    for i in range(75):
        for j in range(3):
            image_pts[i][j]=image_pts[i][j]/image_pts[i][2]

    for i in range(75):
        for j in range(4):
            world_pts[i][j]=world_pts[i][j]/world_pts[i][3]


    # sanity checks
    assert image_pts.shape[0] == world_pts.shape[0]

    # homogeneous coordinates
    assert image_pts.shape[1] == 3 and world_pts.shape[1] == 4
    return image_pts, world_pts


def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)
        
    Returns:
        A: (2*N, 12) matrix A
    """
    N, _ = x.shape
    assert N == X.shape[0]
    A = np.empty((2*N, 12))

    # do cross product to construct A matrix
    for i in range(N):
        A[2*i][0:4] = [0, 0, 0, 0]
        A[2*i][4:8] = -X[i, :]
        A[2*i][8:12] = x[i][1]*X[i, :]
        A[2*i+1][0:4] = X[i, :]
        A[2*i+1][4:8] = [0, 0, 0, 0]
        A[2*i+1][8:12] = -x[i][0]*X[i, :]

    assert A.shape[0] == 2*N and A.shape[1] == 12

    return A


def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A
    
    Returns:
        P: (3, 4) projection matrix P
    """
    # decomposition Matrix A
    P = np.empty((3, 4))
    u, s, v = np.linalg.svd(A)
    P = v[11, :].reshape((3, 4))

    return P

def solve_KR(P):
    """Using th RQ-decomposition find K and R 
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns 
    and doing so correspondingly in R.

    Args:
        P: 3x4 projection matrix.
    
    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix X[i].T
    """

    # RQ-decomposition

    K,R = lg.rq(P[:, 0:3], mode="full")
    # S = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    # if(K[0, 0]<0):
    #     S[0,0]=-1
    # if(K[1, 1]<0):
    #     S[1,1]=-1
    # #if(K[2, 2]<0):
    # k22 = K[2, 2]
    # S[2, 2] = 1/k22
    # #K = K/k22
    # K = K@S
    # #R = K@k22
    #S[2,2] = k22
    #R = S@R
    K22 = K[2, 2]
    if(K[0,0]<0):
        K[0,0] = -K[0, 0]
    if(K[1,1]<0):
        K[1,1] = -K[1, 1]

    K = K/K22
    R = R*K22
    # M = K@R
    # print(P[:, 0:3]-M)

    return K, R

def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix
    
    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    c = np.empty((3, 1))
    # the fourth column of P
    m = P[:, 3].reshape((3, 1))
    c = np.linalg.inv(-P[:, 0:3])@m
    # c = lg.null_space(P)

    return c
