import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use same padding (mode = 'reflect'). Refer docs for further info.

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space, save_img)


def gaussian_filter(image, sigma):
    """
    Given an image, apply a Gaussian filter with the input kernel size
    and standard deviation

    Input
      image: image of size HxW
      sigma: scalar standard deviation of Gaussian Kernel

    Output
      Gaussian filtered image of size HxW
    """
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    # Ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    # TODO implement gaussian filtering of size kernel_size x kernel_size
    # Similar to Corner detection, use scipy's convolution function.
    # Again, be consistent with the settings (mode = 'reflect').
    #x, y = np.meshgrid(np.linspace(0, kernel_size, kernel_size),
    #                   np.linspace(0, kernel_size, kernel_size))
    lin = kernel_size//2
    x, y = np.meshgrid(np.linspace(-(lin), (lin), kernel_size),
                       np.linspace(-(lin), lin, kernel_size))
    kernel = np.exp(-1 * ((x ** 2 + y ** 2) / (2 * (sigma ** 2)))) / (2 * np.pi * sigma**2)
    return scipy.ndimage.convolve(image, kernel, mode = 'reflect')




def main():
    image = read_img('/Users/shauryagunderia/Downloads/eecs442/hw2/cells/001cell.png')
    # import pdb; pdb.set_trace()
    # Create directory for polka_detections

    if not os.path.exists("/Users/shauryagunderia/Downloads/eecs442/hw2/polka_detections"):
        os.makedirs("/Users/shauryagunderia/Downloads/eecs442/hw2/polka_detections")

    # -- TODO Task 8: Single-scale Blob Detection --

    # (a), (b): Detecting Polka Dots
    # First, complete gaussian_filter()
    print("Detecting small polka dots")
    # -- Detect Small Circles

    sigma_1, sigma_2 = 4.3, 3.002
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement
    # calculate difference of gaussians
    #DoG_small = gaussian_filter(gauss_2 - gauss_1, sigma_2-sigma_1)  # to implement
    DoG_small = gauss_1-gauss_2
    #print(DoG_small)
    # visualize maxima
    maxima = find_maxima(DoG_small, k_xy=10, k_s=1)
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          '/Users/shauryagunderia/Downloads/eecs442/hw2/polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     '/Users/shauryagunderia/Downloads/eecs442/hw2/polka_detections/polka_small.png')

    # -- Detect Large Circles
    print("Detecting large polka dots")
    sigma_1, sigma_2 = 8,9
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_large = gauss_2 - gauss_1  # to implement

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          '/Users/shauryagunderia/Downloads/eecs442/hw2/polka_detections/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     '/Users/shauryagunderia/Downloads/eecs442/hw2/polka_detections/polka_large.png')


    # # -- TODO Task 9: Cell Counting --
    print("Detecting cells")
    x, y = np.meshgrid(np.linspace(-(1), (1), 5),
                       np.linspace(-(1), 1, 5))
    sig = .5
    kernel = np.exp(-1 * ((x ** 2 + y ** 2) / (2 * (sig ** 2)))) / (2 * np.pi * sig ** 2)
    # Detect the cells in any four (or more) images from vgg_cells
    # Create directory for cell_detections
    if not os.path.exists("/Users/shauryagunderia/Downloads/eecs442/hw2/cell_detections"):
        os.makedirs("/Users/shauryagunderia/Downloads/eecs442/hw2/cell_detections")
    Sx = np.array(([1, 0, -1], [2, 0, -1], [1, 0, -1]))
    Sy = Sx.T
    sigma_1, sigma_2 = 3.3,4.5
    image = gaussian_filter(image, sig)
    edgex = scipy.ndimage.convolve(image, Sx, mode='constant')
    edgey = scipy.ndimage.convolve(image, Sy, mode = 'constant')
    deriv = (edgex**2 + edgey**2)**0.5
    save_img(deriv, "./cell_detections/cell01deriv.png")
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    edgegx1 = scipy.ndimage.convolve(gauss_1, Sx, mode='constant')
    edgegy1 = scipy.ndimage.convolve(gauss_1, Sy, mode='constant')
    derivg1 = (edgegx1 ** 2 + edgegy1 ** 2) ** 0.5
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement
    edgegx2 = scipy.ndimage.convolve(gauss_2, Sx, mode='constant')
    edgegy2 = scipy.ndimage.convolve(gauss_2, Sy, mode='constant')
    derivg2 = (edgegx2 ** 2 + edgegy2 ** 2) ** 0.5
    # calculate difference of gaussians
    gauss_1 = scipy.ndimage.convolve(gauss_1, np.array(([0, -1, 0], [-1, 1.1, -1], [0, -1, 0])), mode='constant')
    gauss_1 = gaussian_filter(gauss_1, 1)
    gauss_2 = scipy.ndimage.convolve(gauss_2, np.array(([0, -1, 0], [-1, 1.1, -1], [0, -1, 0])), mode='constant')
    gauss_2 = gaussian_filter(gauss_2, 1)
    #g1 = scipy.ndimage.convolve(gauss_1, np.array(([-1, -1, -1], [-1, 9, -1], [-1, -1, -1])), mode='constant')
    #g2 = scipy.ndimage.convolve(gauss_2, np.array(([-1, -1, -1], [-1, 9, -1], [-1, -1, -1])), mode='constant')
    DoG_large = gauss_1-gauss_2 # to implement
    DoG_large = gaussian_filter(DoG_large, 1.8)
    DoG_large = scipy.ndimage.convolve(DoG_large, np.array(([.5,.5,.5],[.5,.9,.5],[.5,.5,.5])))
    #dogx = scipy.ndimage.convolve(DoG_large, Sx, mode='constant')

    # visualize maxima
    # Value of k_xy is a sugguestion; feel
    # free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=13, k_s=3)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          '/Users/shauryagunderia/Downloads/eecs442/hw2/cell_detections/Cell01_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     '/Users/shauryagunderia/Downloads/eecs442/hw2/cell_detections/Cell01.png')




if __name__ == '__main__':
    main()
