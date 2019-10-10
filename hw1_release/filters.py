"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flipud(np.fliplr(kernel))
    for m in range(Hi):
        for n in range(Wi):
            s = 0
            for i in range(Hk):
                for j in range(Wk):
                    l = m + i - Hk//2
                    k = n + j - Wk//2
                    # Boundary condition
                    if l < 0 or k < 0 or l > (Hi-1) or k > (Wi-1):
                        p = 0
                    else:
                        p = image[l, k]
                    s += p * kernel[i, j] 
            out[m, n] = s
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + pad_height * 2, W + pad_width * 2))
    out[pad_height:-pad_height, pad_width:-pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flipud(np.fliplr(kernel))
    pad_h, pad_w = Hk//2, Wk//2
    img_pad = image.copy()
    img_pad = zero_pad(img_pad, pad_h, pad_w)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = (img_pad[i: i+Hk, j:j+Wk] * kernel).sum()
    ### END YOUR CODE
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flipud(np.fliplr(kernel))
    pad_h, pad_w = Hk//2, Wk//2
    img_pad = image.copy()
    img_pad = zero_pad(img_pad, pad_h, pad_w)

    for i in range(Hi):
        for j in range(Wi):
            if j == 0:
                m = img_pad[i: i+Hk, j:j+Wk] * kernel
                out[i, j] = m.sum()
                neg = m[:, 0].sum()
            else:
                m = kernel[:, 2] * img_pad[i: i+Hk, j+Wk-1]
                out[i, j] = out[i, j-1] - neg + m.sum()
                neg = (kernel[:, 0] * img_pad[i: i+Hk, j]).sum()
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    
    Hk, Wk = g.shape
    Hi, Wi = f.shape
    out = np.zeros((Hi, Wi))

    
    pad_h, pad_w = Hk//2, Wk//2
    img_pad = f.copy()
    img_pad = zero_pad(img_pad, pad_h, pad_w)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = (img_pad[i: i+Hk, j:j+Wk] * g).sum()
    ### END YOUR CODE
    return out
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = g-g.mean()
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = (g - g.mean()) / g.std()
    Hk, Wk = g.shape
    Hi, Wi = f.shape
    out = np.zeros((Hi, Wi))

    
    pad_h, pad_w = Hk//2, Wk//2
    img_pad = f.copy()
    img_pad = zero_pad(img_pad, pad_h, pad_w)

    for i in range(Hi):
        for j in range(Wi):
            m = img_pad[i: i+Hk, j:j+Wk]
            m = (m - m.mean()) / m.std()
            
            out[i, j] = (m * g).sum()
    ### END YOUR CODE

    return out
