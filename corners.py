import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import cv2
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img


def corner_score(image, u, v, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    output = np.roll(image, (u, v), axis=(1,0))
    roll = (output - image)**2
    output = scipy.ndimage.convolve(roll, np.ones(window_size), mode='constant')
    return output


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = scipy.ndimage.convolve(image, kernel_x, mode='constant')
    Iy = scipy.ndimage.convolve(image, kernel_y, mode='constant')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    x, y = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))   #play with this
    #x, y = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
    sdev = 1.3
    kernel_gaussian = np.exp(-1 * ((x ** 2 + y ** 2) / 2 * (sdev ** 2))) / (2 * np.pi * (sdev ** 2))
    Ixx = scipy.ndimage.convolve(Ixx, kernel_gaussian, mode='constant')
    Iyy = scipy.ndimage.convolve(Iyy, kernel_gaussian, mode='constant')
    Ixy = scipy.ndimage.convolve(Ixy, kernel_gaussian, mode='constant')
    detM = Ixx*Iyy - Ixy**2
    traceM = Ixx + Iyy
    response = detM - 0.04*traceM**2
    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 5: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    #u, v, W = -5,0,(5,5)

    score = corner_score(img, -5, 0)
    save_img(score, "./feature_detection/corner_score3.png")

    # (c): No Code

    # -- TODO Task 6: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    #img2 = cv2.imread('./grace_hopper.png', cv2.IMREAD_GRAYSCALE)
    harris_corners = harris_detector(img)
    #hcFunc = cv2.cornerHarris(img2, 5, 3, 0.05)
    save_img(harris_corners, "./feature_detection/harris_response.png")
    #save_img(hcFunc, "./feature_detection/hrFunc.png")
    plt.imshow(harris_corners, cmap='jet', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    main()
