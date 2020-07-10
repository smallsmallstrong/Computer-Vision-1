import numpy as np

def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """

    #
    # Your code goes here
    #
    cost_ssd = np.sum(np.power(patch1-patch2,2))

    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """

    #
    # Your code goes here
    #
    m = patch1.shape[0]
    length = m*m
    wl = patch1.reshape(length,) - np.mean(patch1)
    wr = patch2.reshape(length,) - np.mean(patch2)
    cost_nc = wl.T@wr/(np.linalg.norm(wl)*np.linalg.norm(wr))
    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        input_disparity: input disparity as an integer value        
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape 

    #
    # Your code goes here
    #
    m = patch1.shape[0]
    cost_val = 1/(m*m)*cost_ssd(patch1,patch2)+alpha*cost_nc(patch1,patch2)
    
    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image
    
    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'
        
    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """

    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    #
    # Your code goes here
    #
    pad_width = int((window_size-1)/2)
    padded_img = input_img.copy()
    padded_img = np.pad(padded_img,pad_width,padding_mode)
    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2 
    assert padded_img_r.ndim == 2 
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    #
    # Your code goes here
    #
    #disparity = padded_img_l.copy()

    nr,nc = padded_img_l.shape
    origw = nc - window_size+1
    origh = nr - window_size+1
    disparity = np.empty((origh,origw))
    for i in range(origh):
        for j in range(origw):
            tempwl = padded_img_l[i:i+window_size,j:j+window_size]
            tempdisparity = 0
            costmintemp = 1000000
            begin = j-max_disp
            if(begin < 0):
                begin = 0
            for k in range(begin,j+1):
                tempwr = padded_img_r[i:i+window_size,k:k+window_size]
                tempcost = cost_function(tempwl,tempwr,alpha)
                if(tempcost<costmintemp):
                    costmintemp = tempcost
                    tempdisparity = j-k

            disparity[i,j] = tempdisparity


    assert disparity.ndim == 2
    return disparity

def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:
    
    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape

    #
    # Your code goes here
    #
    h,w = disparity_gt.shape
    aepe = np.sum(np.abs(disparity_gt-disparity_res))/(h*w)

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #

    alpha = -0.01
    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer 
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
        
        Q1. [1] is better for estimating disparity values on sharp objects and object boundaries
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [2] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [1] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [2].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """

        return (1, 2, 1, 2)
