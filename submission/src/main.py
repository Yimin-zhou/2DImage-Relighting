import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from sobel_normal import calculate_normal_sobel
from depth_normal import calculate_depth_relative, calculate_depth_face, depth_to_normal
from full_body_normal import calculate_normal
from albedo import calculate_albedo
from hdr_filtering import prefilter_cube_map, gaussian
from shading import calculate_diffuse, apply_light_map

# set path
dirname = os.path.dirname(os.path.abspath(__file__))
output_image_dir= dirname + '/outputs/'
input_image_path = dirname + '/inputs/'
input_mask_path = dirname + '/inputs/'
cube_map_path = dirname + '/inputs/cubemap/red/'
cube_out_path = dirname + '/outputs/cubemap/'
cubemap_names = ['px', 'nx', 'py', 'ny', 'pz', 'nz']

img_szie = 1024
input_img_name = 'input2.jpg'
input_image = np.float32(Image.open(input_image_path + input_img_name))
input_image_mask = np.float32(Image.open(input_mask_path + 'mask_' + input_img_name))
input_image = resize(input_image, (img_szie, img_szie))
input_image_mask = resize(input_image_mask, (img_szie, img_szie))


def main():
    ############################### Step 1 surface normal ########################################
    print("1. Calculating surface normal...")
    # (Alternative surface normal method 1)
    # calculate normal map 
    sobel_normal = calculate_normal_sobel(input_image, img_szie)
    sobel_normal_output = Image.fromarray(np.uint8(sobel_normal))
    if sobel_normal_output.mode != 'RGB':
        sobel_normal_output = sobel_normal_output.convert('RGB')
    sobel_normal_output.save(output_image_dir + 'sobel_normal_' + input_img_name )
    
    # (Alternative surface normal method 2)
    # step 1 calculate depth map
    # - get the depth map
    depth = calculate_depth_face(input_image, img_szie)
    # depth = calculate_depth_relative(input_image) # relative depth estimation
    depth = resize(depth, (img_szie, img_szie)) # have to resize depth again
    depth = np.multiply(depth, 255)
    depth_output = Image.fromarray(depth)
    if depth_output.mode != 'RGB':
        depth_output = depth_output.convert('RGB')
    depth_output.save(output_image_dir + 'depth_' + input_img_name )
    # - derive normal map from depth map
    depth_normal = depth_to_normal(input_image, depth, img_szie)
    normal_output = Image.fromarray(np.uint8(depth_normal))
    if normal_output.mode != 'RGB':
        normal_output = normal_output.convert('RGB')
    normal_output.save(output_image_dir + 'normal_' + input_img_name )
    
    # (Alternative surface normal method 3)
    # calculate normal map 
    # calculate_normal(input_image_path, (input_image_path + 'input1.jpg'), output_image_dir)
    
    surface_normal = depth_normal
    print("1. Done surface normal!")
    
    ############################### Step 2 albedo ########################################
    # step 2 calculate albedo map
    print("2. Calculating albedo normal...")
    albedo = calculate_albedo(input_image, img_szie)
    albedo_ouput = Image.fromarray(np.uint8(albedo))
    if albedo_ouput.mode != 'RGB':
        albedo_ouput = light_ouput.convert('RGB')
    albedo_ouput.save(output_image_dir + 'albedo_' + input_img_name )
    print("2. Done albedo!")
    
    ############################### Step 3 HDR to lightmap ########################################
    print("3. Proccessing Cubemap to lightmap...")
    # step 3 prefiltering Cubemap
    px = np.float32(Image.open(cube_map_path + 'px.png'))
    nx = np.float32(Image.open(cube_map_path + 'nx.png'))
    py = np.float32(Image.open(cube_map_path + 'py.png'))
    ny = np.float32(Image.open(cube_map_path + 'ny.png'))
    pz = np.float32(Image.open(cube_map_path + 'pz.png'))
    nz = np.float32(Image.open(cube_map_path + 'nz.png'))
    cube_maps = [px, nx, py, ny, pz, nz]
    filtered_cube_maps = prefilter_cube_map(cube_maps, 150)
    i = 0
    for filtered in filtered_cube_maps:
        filtered_output = Image.fromarray(np.uint8(filtered))
        if filtered_output.mode != 'RGB':
            filtered_output = filtered_output.convert('RGB')
        filtered_output.save(cube_out_path + f"{cubemap_names[i]}.jpg")
        i = i + 1
        
    print("3. Done Cubemap to lightmap!")
    
    # step 4 light map
    print("3. Proccessing light map...")
    light = calculate_diffuse(filtered_cube_maps, surface_normal, img_szie)
    light = gaussian(light, sigma=100)
    light_ouput = Image.fromarray(np.uint8(light))
    if light_ouput.mode != 'RGB':
        light_ouput = light_ouput.convert('RGB')
    light_ouput.save(output_image_dir + 'light_' + input_img_name )
    print("3. Done light map!")
    
    # step 5 mask images
    print("5. Masking out backgroud...")
    light = light * (input_image_mask / 255)
    albedo = albedo * (input_image_mask / 255)
    mask_ouput = Image.fromarray(np.uint8(albedo))
    if mask_ouput.mode != 'RGB':
        mask_ouput = light_ouput.convert('RGB')
    mask_ouput.save(output_image_dir + 'mask_' + input_img_name )
    print("5. Done get rid of backgroud!")
    
    # step 6 compositing
    print("6. Compositing final result...")
    # repalce background
    background = filtered_cube_maps[2] * (1 - (input_image_mask / 255))
    final= apply_light_map(albedo, light) * 255 + background 
    final_ouput = Image.fromarray(np.uint8(final))
    if final_ouput.mode != 'RGB':
        final_ouput = light_ouput.convert('RGB')
    final_ouput.save(output_image_dir + 'final_' + input_img_name )
    
    # background replacement
    print("6. Done final result!")

if __name__ == '__main__':
    main()

