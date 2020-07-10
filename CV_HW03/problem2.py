import numpy as np
import math
import matplotlib.pyplot as plt


def load_pts_features(path):
    """ Load interest points and SIFT features.

    Args:
        path: path to the file pts_feats.npz
    
    Returns:
        pts: coordinate points for two images;
             an array (2,) of numpy arrays (N1, 2), (N2, 2)
        feats: SIFT descriptors for two images;
               an array (2,) of numpy arrays (N1, 128), (N2, 128)
    """

    #
    # Your code here
    #
    file = np.load(path, allow_pickle='True')
    pts, feats = file['pts'], file['feats']
    return pts, feats


def min_num_pairs():
    return 4


def pickup_samples(pts1, pts2):
    """ Randomly select k corresponding point pairs.
    Note that here we assume that pts1 and pts2 have 
    been already aligned: pts1[k] corresponds to pts2[k].

    This function makes use of min_num_pairs()

    Args:
        pts1 and pts2: point coordinates from Image 1 and Image 2
    
    Returns:
        pts1_sub and pts2_sub: N_min randomly selected points 
                               from pts1 and pts2
    """

    #
    # Your code here
    #
    N_min = min_num_pairs()
    n, _ = pts1.shape
    pts1_sub = np.empty((N_min, 2))
    pts2_sub = np.empty((N_min, 2))

    def sample_one_unique():
        listindex = []
        while len(listindex) < 4:
            tempindex = np.random.randint(0, n, size=1)[0]
            if tempindex not in listindex:
                listindex.append(tempindex)
        indexarray = np.array(listindex)

        for i in range(N_min):
            pts1_sub[i, :] = pts1[indexarray[i], :]
            pts2_sub[i, :] = pts2[indexarray[i], :]

    def resample_condition():# confirm linear independence
        j = 0
        A = np.empty((2 * N_min, 9))
        for i in range(N_min):
            x = pts1_sub[i, 0]
            y = pts1_sub[i, 1]
            x1 = pts2_sub[i, 0]
            y1 = pts2_sub[i, 1]
            A[j, :] = [0, 0, 0, x, y, 1, -x * y1, -y * y1, -y1]
            A[j + 1, :] = [-x, -y, -1, 0, 0, 0, x * x1, y * x1, x1]
            j = j + 2
        if(np.linalg.matrix_rank(A)<8):
            return 1
        else:
            return 0

    sample_one_unique()
    while(resample_condition()==1):
        sample_one_unique()

    return pts1_sub, pts2_sub


def compute_homography(pts1, pts2):
    """ Construct homography matrix and solve it by SVD

    Args:
        pts1: the coordinates of interest points in img1, array (N, 2)
        pts2: the coordinates of interest points in img2, array (M, 2)
    
    Returns:
        H: homography matrix as array (3, 3)
    """

    #
    # Your code here
    #

    def preprocess(points):  # scale and shift points to be in [-1..1]
        tempnorm = np.linalg.norm(points,axis=1)
        xmax = np.amax(tempnorm) / 2
        tx, ty = np.mean(points, axis=0)
        temph, tempw = points.shape
        newpoints = np.empty((temph, 2))
        for i in range(temph):
            newpoints[i, 0] = 1 / xmax * points[i, 0] - tx / xmax
            newpoints[i, 1] = 1 / xmax * points[i, 1] - ty / xmax

        return newpoints, xmax, tx, ty

    pts1_sub, pts2_sub = pts1,pts2#pickup_samples(pts1, pts2)
    pts1_sub, s1, tx1, ty1 = preprocess(pts1_sub)
    pts2_sub, s2, tx2, ty2 = preprocess(pts2_sub)
    n, _ = pts1_sub.shape

    A = np.empty((2 * n, 9))
    j = 0
    for i in range(n):
        x = pts1_sub[i, 0]
        y = pts1_sub[i, 1]
        x1 = pts2_sub[i, 0]
        y1 = pts2_sub[i, 1]
        A[j, :] = [0, 0, 0, x, y, 1, -x * y1, -y * y1, -y1]
        A[j + 1, :] = [-x, -y, -1, 0, 0, 0, x * x1, y * x1, x1]
        j = j+2

    u, s, v = np.linalg.svd(A)
    H = v[8,:].reshape((3, 3))

    #tempb = A@v[8,:].T
    #tempa = transform_pts(pts1_sub,H)

    T1 = np.array([[1 / s2, 0, -tx2 / s2],
                   [0, 1 / s2, -ty2 / s2],
                   [0, 0, 1]])
    T = np.array([[1 / s1, 0, -tx1 / s1],
                  [0, 1 / s1, -ty1 / s1],
                  [0, 0, 1]])

    H = np.linalg.inv(T1) @ H @ T

    #transform_points = transform_pts(pts1, H)

    return H


def transform_pts(pts, H):
    """ Transform pst1 through the homography matrix to compare pts2 to find inliners

    Args:
        pts: interest points in img1, array (N, 2)
        H: homography matrix as array (3, 3)
    
    Returns:
        transformed points, array (N, 2)
    """

    #
    # Your code here
    #
    h, w = pts.shape
    temppts = np.ones((h, 3))
    temppts[:, 0:2] = pts
    transform_points = H @ temppts.T  # 3*h

    for i in range(h):
        transform_points[:, i] = transform_points[:, i] / transform_points[2, i]

    return transform_points.T[:, 0:2]


def count_inliers(H, pts1, pts2, threshold=5):
    """ Count inliers
        Tips: We provide the default threshold value, but you’re free to test other values
    Args:
        H: homography matrix as array (3, 3)
        pts1: interest points in img1, array (N, 2)
        pts2: interest points in img2, array (N, 2)
        threshold: scale down threshold
    
    Returns:
        number of inliers
    """
    transform_points = transform_pts(pts1, H)
    n, _ = pts1.shape
    count = 0
    for i in range(n):
        if(np.linalg.norm((transform_points[i, :] - pts2[i, :]))<threshold):
            count = count+1

    return count


def ransac_iters(w=0.5, d=min_num_pairs(), z=0.99):
    """ Computes the required number of iterations for RANSAC.

    Args:
        w: probability that any given correspondence is valid
        d: minimum number of pairs
        z: total probability of success after all iterations
    
    Returns:
        minimum number of required iterations
    """
    k = math.log2(1 - z) / math.log2(1 - math.pow(w, d))
    return int(k)


def ransac(pts1, pts2):
    """ RANSAC algorithm

    Args:
        pts1: matched points in img1, array (N, 2)
        pts2: matched points in img2, array (N, 2)
    
    Returns:
        best homography observed during RANSAC, array (3, 3)
    """

    #
    # Your code here
    #
    k = ransac_iters()
    best_H = None
    max_count = -1
    for i in range(k):
        sub_pts1, sub_pts2 = pickup_samples(pts1, pts2)
        H = compute_homography(sub_pts1, sub_pts2)
        count = count_inliers(H, pts1, pts2)
        if (count > max_count):
            max_count = count
            best_H = H

    return best_H


def find_matches(feats1, feats2, rT=0.8):
    """ Find pairs of corresponding interest points with distance comparsion
        Tips: We provide the default ratio value, but you’re free to test other values

    Args:
        feats1: SIFT descriptors of interest points in img1, array (N, 128)
        feats2: SIFT descriptors of interest points in img1, array (M, 128)
        rT: Ratio of similar distances
    
    Returns:
        idx1: list of indices of matching points in img1
        idx2: list of indices of matching points in img2


    """

    idx1 = []
    idx2 = []

    #
    # Your code here
    #

    n, _ = feats1.shape
    m, _ = feats2.shape

    for i in range(n):
        minimum0 = 100000000
        minimum1 = 100000000
        index = None
        for j in range(m):
            tempd = np.linalg.norm(feats1[i, :] - feats2[j, :])
            if (tempd < minimum1):
                if (tempd < minimum0):
                    minimum0 = tempd
                    index = j
                else:
                    minimum1 = tempd

        if (minimum0 / minimum1 < rT):  #
            idx1.append(i)
            idx2.append(index)

    return idx1, idx2


def final_homography(pts1, pts2, feats1, feats2):
    """ re-estimate the homography based on all inliers

    Args:
       pts1: the coordinates of interest points in img1, array (N, 2)
       pts2: the coordinates of interest points in img2, array (M, 2)
       feats1: SIFT descriptors of interest points in img1, array (N, 128)
       feats2: SIFT descriptors of interest points in img1, array (M, 128)
    
    Returns:
        ransac_return: refitted homography matrix from ransac fucation, array (3, 3)
        idxs1: list of matched points in image 1
        idxs2: list of matched points in image 2
    """
    #
    # Your code here
    #

    threshold = 5
    idxs1, idxs2 = find_matches(feats1, feats2)
    n = len(idxs1)
    newpts1 = np.empty((n, 2))
    newpts2 = np.empty((n, 2))
    for i in range(n):
        newpts1[i, :] = pts1[idxs1[i], :]
        newpts2[i, :] = pts2[idxs2[i], :]

    best_H = ransac(newpts1, newpts2)

    transform_points = transform_pts(newpts1, best_H)
    inliersindex = []
    count = 0
    for i in range(n):
        if (np.linalg.norm((transform_points[i, :] - newpts2[i, :])) < threshold):
            inliersindex.append(i)
            count = count + 1

    inliers1 = np.empty((count, 2))
    inliers2 = np.empty((count, 2))
    for i in range(count):
        inliers1[i,:] = newpts1[inliersindex[i],:]
        inliers2[i,:] = newpts2[inliersindex[i],:]

    ransac_return = compute_homography(inliers1,inliers2)

    return ransac_return, idxs1, idxs2
