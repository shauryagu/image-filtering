import os

import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt

from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output = np.empty((int(image.shape[0]/16) + int(image.shape[1]/16), 16, 16))
    num = 0
    for i in range(int(image.shape[0]/16)):
        for j in range(int(image.shape[1]/16)):
            output[num] = image[(i*16):(i+1)*16, j*16:(j+1)*16]
            output[num] = output[num] - np.mean(output[num])
            output[num] = output[num]/np.std(output[num])
        num = num+1
    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    kernel = kernel[::-1, ::-1]
    p = int(kernel.shape[0]/2)
    padding = np.zeros((image.shape[0] + p, image.shape[1] + p))
    output = np.zeros((image.shape[0], image.shape[1]))
    padding[p:(image.shape[0] + p), p:(image.shape[1] + p)] = image
    for x in range(padding.shape[0]-2*p):
        for y in range(padding.shape[1]-2*p):
            for fx in range(kernel.shape[0]):
                for fy in range(kernel.shape[1]):
                    output[x, y] = output[x, y] + padding[x+fx, y+fy]*kernel[fx, fy] #check this
    return output



def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([[-1],[0], [1]]).T  # 1 x 3
    ky = np.array([[-1],[0], [1]])  # 3 x 1
    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    Sx = np.array(([1,0,-1],[2,0,-1],[1,0,-1]))
    Sx = Sx[::-1, ::-1]
    Sy = Sx.T
    Sy = Sy[::-1, ::-1]
    Gx = convolve(image, Sx)
    Gy = convolve(image, Sy)
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)

    return Gx, Gy, grad_magnitude




def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = np.hstack((patches[0], patches[1], patches[2]))
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.

    x, y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
    sdev = 0.572
    kernel_gaussian = np.exp(-1*((x**2+y**2)/(2*(sdev**2))))/(2*np.pi*(sdev**2))
    kernel_gaussian = kernel_gaussian[::-1, ::-1]
    filtered_gaussian = convolve(img, kernel_gaussian)
    #filtered_gaussian = scipy.ndimage.convolve(img, kernel_gaussian, mode='constant')
    save_img(filtered_gaussian, "./gaussian_filter/q2scale_gaussian.png")

    # (d), (e): No code

    # (f): Complete edge_detection()
    '''
    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    
    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code
    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    data = np.load('log1d.npz')
    plt.plot(data['log50'])
    plt.show()
    plt.plot(data['gauss53'] - data['gauss50'])
    plt.show()
    print("LoG Filter is done. ")
    '''
if __name__ == "__main__":
    main()
