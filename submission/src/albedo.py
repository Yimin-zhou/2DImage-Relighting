import numpy as np

def ps_soft_light_blend(img1, img2, size):
    # soft light blend, same as in photoshop
    result = np.full_like(img1, 0., dtype=float)
    base = img1 / 255.
    blend = img2 / 255.
    for y in range(size):
        for x in range(size):
            for c in range(3):
                if(blend[y, x, c] < 0.5):
                    result[y, x, c] = 2.0 * base[y, x, c] * blend[y, x, c] + base[y, x, c] * base[y, x, c] * (1.0 - 2.0 * blend[y, x, c])
                else:
                    result[y, x, c] = np.sqrt(base[y, x, c]) * (2.0 * blend[y, x, c] - 1.0) + 2.0 * base[y, x, c] * (1.0 - blend[y, x, c])
    return result * 255.

def calculate_albedo(img, size):
    albedo = np.full_like(img, 0., dtype=float)
    # desaturate
    for y in range(size):
        for x in range(size):
            albedo[y, x, :] = np.mean(img[y, x, :])
    # soft light blend
    albedo = ps_soft_light_blend(img, albedo, size)
            
    return albedo