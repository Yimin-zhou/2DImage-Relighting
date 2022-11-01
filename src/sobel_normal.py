import torch
import numpy as np
import math
from vendors.depth_predict import face_depth

def get_intensity(img, size):
    intensity = np.zeros((size, size), dtype=float)
    for y in range(size):
        for x in range(size):
            intensity[y, x] = np.mean(img[y, x, :])
    # remap to 0-1
    return intensity / 255.
    
def clamp(i, size):
    return max(0, min(i, size-1))

def calculate_normal_sobel(img, size, strength=4.):
    result = np.full_like(img, 0., dtype=float)
    inten = get_intensity(img, size)
    for y in range(size):
        for x in range(size):
            tl = inten[clamp(y - 1, size), clamp(x - 1, size)]
            tp = inten[clamp(y - 1, size), clamp(x, size)]
            tr = inten[clamp(y - 1, size), clamp(x + 1, size)]
            l = inten[clamp(y, size), clamp(x - 1, size)]
            r = inten[clamp(y, size), clamp(x + 1, size)]
            br = inten[clamp(y + 1, size), clamp(x + 1, size)]
            bt = inten[clamp(y + 1, size), clamp(x, size)]
            bl = inten[clamp(y + 1, size), clamp(x - 1, size)]
            
            # sobel
            dx = (tr + 2. * r + br) - (tl + 2. * l + bl);
            dy = (bl + 2. * bt + br) - (tl + 2. * tp + tr);
            dz = 1. / strength;
            # normalize
            v = np.array([dx, dy, dz])
            v = v / np.linalg.norm(v) # -1 - 1 range
            result[y ,x :] = (v / 2. + 0.5) * 255. # map to 0 - 255.
            
    return result
    