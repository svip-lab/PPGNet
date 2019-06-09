from numba import jit, float32, int32
import numpy as np    
    

@jit(float32[:, :](float32[:, :], float32[:, :], int32[:, :], int32[:, :], float32), nopython=True, fastmath=True)
def apply_gaussian(accumulate_confid_map, centers, xx, yy, sigma):
    for i in range(len(centers)):
        center = centers[i]
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= 4.6052
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        accumulate_confid_map += cofid_map
    return accumulate_confid_map

def gen_gaussian_map(centers, shape, sigma):
    centers = np.float32(centers)
    sigma = np.float32(sigma)
    accumulate_confid_map = np.zeros(shape, dtype=np.float32)
    y_range = np.arange(accumulate_confid_map.shape[0], dtype=np.int32)
    x_range = np.arange(accumulate_confid_map.shape[1], dtype=np.int32)
    xx, yy = np.meshgrid(x_range, y_range)

    accumulate_confid_map = apply_gaussian(accumulate_confid_map, centers, xx, yy, sigma)
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    
    return accumulate_confid_map
