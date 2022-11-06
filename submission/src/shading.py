import numpy as np

def apply_light_map(albedo, light):
    return light / 255 * albedo / 255 * 1.2

def calculate_diffuse(cub_maps, surface_normal, size):
    light = np.full_like(surface_normal, 0., dtype=float)
    normal_coordinate = (surface_normal / 255.  + 0.5) * 2. # map normal back to -1 - 1
    # cubemap_names = ['px'0, 'nx'1, 'py'2, 'ny'3, 'pz'4, 'nz'5]
    for y in range(size):
        for x in range(size):
            # sample cubemap base on the normal's the largest magnitude coordinate component
            normal_x = normal_coordinate[y, x, 0]
            normal_y = normal_coordinate[y, x, 1]
            normal_z = normal_coordinate[y, x, 2]
            axis_max = max(max(abs(normal_x), abs(normal_y)), abs(normal_z))
            # determine which face to sample
            if abs(normal_x) == axis_max:
                which_face = 0 if normal_coordinate[y, x, 0] > 0 else 1
                # normalize & remap 0-1 the coordinate used to sample map
                l = cub_maps[which_face][int((normal_y / axis_max + 1) / 2)][int((normal_z / axis_max + 1) / 2)]
            if abs(normal_y) == axis_max:
                which_face = 2 if normal_coordinate[y, x, 0] > 0 else 3
                l = cub_maps[which_face][int((normal_x / axis_max + 1) / 2)][int((normal_z / axis_max + 1) / 2)]
            if abs(normal_z) == axis_max:
                which_face = 4 if normal_coordinate[y, x, 0] > 0 else 5
                l = cub_maps[which_face][int((normal_x / axis_max + 1) / 2)][int((normal_y / axis_max + 1) / 2)]
            # print(l)
            light[y, x, :] = l[:]
    # samped cube maps are in range 0-255
    return light
