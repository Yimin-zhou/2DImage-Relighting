from scipy import ndimage
import numpy as np
import cv2


def gaussian_kernel(s, sigma):
    ax = np.linspace(-(s - 1) / 2., (s - 1) / 2., s)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return kernel / np.sum(kernel)

    
def gaussian(img, sigma=50, k_size=100):
    result = img
    gkernel = gaussian_kernel(k_size, sigma)
    for i in range(3):
        result[:, :, i] = cv2.filter2D(result[:, :, i], -1, gkernel) # only for convolution
        
    return result

def prefilter_cube_map(cubmaps, level):
    result = []
    for env_map in cubmaps:
        result.append(gaussian(env_map, sigma=level))
    return result