import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve2d

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []

    #
    # TODO
    #

    # load face images and face features as two lists of numpy arrays
    root = path
    f_images = 'facial_images'
    f_features = 'facial_features'
    for i in range(5):
        j = i
        img = os.path.join(root, f_images, '0'+str(i+1)+'.pgm')
        img_temp = plt.imread(img)
        imgs.append(img_temp)
        if(i == 2): j = 0
        if(i == 3): j = 2
        if(i == 4): j = 1
        feat = os.path.join(root, f_features, '0'+str(i)+'_'+'0'+str(j)+'.pgm')
        feat_temp = plt.imread(feat)
        feats.append(feat_temp)

    # plt.imshow(imgs[0])
    # plt.show()
    # print(type(imgs))
    return imgs, feats


def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    #
    # TODO
    #

    kernel = np.zeros([fsize, fsize])
    center = fsize//2
    s = 2*(sigma**2)
    sum = 0
    for i in range(fsize):
        for j in range(fsize):
            x = i-center
            y = j-center
            kernel[i, j] = np.exp(-(x**2+y**2)/s)
            sum += kernel[i, j]

    gauss = kernel/sum

    return gauss


def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    #
    # TODO
    #

    h, w = x.shape
    downsample = np.copy(x)
    # get all the row and all even list
    # [::factor],[factor::]
    list_s = downsample[:, range(1, w, factor)]# range(start, stop, step)
    # get all the even row from list above
    row_s = list_s[range(1, h, factor), :]
    sample = row_s

    return sample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    down_image = np.copy(img)
    gauss = gaussian_kernel(fsize, sigma)
    GP = []
    #
    # TODO
    #
    # for i in range(nlevels-1):
    #     image = convolve2d(gauss, image)
    #     image_sample = downsample_x2(image, 2)
    #     GP.append(image)
    while True:
        GP.append(down_image)
        nlevels=nlevels-1
        if(nlevels>0):
            image = convolve2d(down_image,gauss,boundary='symm',mode='same')
            down_image = downsample_x2(image)
        else:
            break


    return GP


def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    distance = None
    #
    # TODO
    #
    # this value actually is cosine
    # distance = v1@v2.T/(np.linalg.norm(v1,ord=2)*np.linalg.norm(v2,ord=2))
    distance = np.dot(v1, v2) / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2))
    # print(distance)
    return distance


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    min_score = -1
    img_temp = np.array(img)
    feat_temp = np.array(feat)

    h_image, w_image = img_temp.shape
    h_feat, w_feat = feat_temp.shape

    v1 = feat_temp.reshape((-1,))
    # define window size from feature
    (h_window, w_window) = (h_feat, w_feat)
    # if dimension doesn't matches,use symmetric padding
    img_pad = np.pad(img_temp, ((0, max(h_feat-h_image, 0)), (0, max(w_feat-w_image, 0))), mode='symmetric')
    h_image_pad, w_img_pad = img_pad.shape
    # list
    for y in range(0, h_image_pad - h_window+1, step):
        # row
        for x in range(0, w_img_pad - w_window+1, step):
            # use dot product to calculate the distance between v1 and v2
            v2 = img_pad[y:y + h_window, x:x + w_window].reshape((-1,))
            score = template_distance(v1, v2)
            if(score > min_score):
                min_score = score
    return min_score


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (1,  5)  # TODO


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays

    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    (score, g_im, feat) = (None, None, None)
    #
    # TODO
    #
    for feat in feats:
        min_score = -1
        for img in imgs:
            # calculate each img from level 3
            pyramids = gaussian_pyramid(img, 3, 5, 1.4)
            for pyramid in pyramids:
                # use each pyramid to calculate distance
                score = sliding_window(pyramid, feat, 1)
                if(min_score < score):
                    min_score = score
                    g_im = pyramid

        match.append((min_score, g_im, feat))


    return match