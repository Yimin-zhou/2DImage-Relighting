from scipy import ndimage

def gaussian(img, level):
    # TODO
    ...

def prefilter_cube_map(cubmaps):
    result = []
    level = 100
    for env_map in cubmaps:
        # TODO (David) replace this to our gaussian function
        result.append(ndimage.gaussian_filter(env_map, sigma=(level,level,0)))
        # result.append(gaussian(env_map, level))
    return result