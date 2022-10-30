import os
import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize
from sobel_normal import calculate_normal_sobel
from depth_normal import calculate_depth_relative, calculate_depth_face, depth_to_normal
from surface_normal import calculate_normal
from albedo import calculate_albedo
from hdr_filtering import prefilter_cube_map
from shading import calculate_diffuse
from scipy import ndimage

# set path
dirname = os.path.dirname(os.path.abspath(__file__))
output_image_dir= dirname + '/outputs/'
input_image_path = dirname + '/inputs/'
input_mask_path = dirname + '/inputs/'
cube_map_path = dirname + '/inputs/cubemap/pink_sunrise/'
cube_out_path = dirname + '/outputs/cubemap/'
cubemap_names = ['px', 'nx', 'py', 'ny', 'pz', 'nz']

img_szie = 1024
input_img_name = 'input5.jpg'
input_image = np.float32(Image.open(input_image_path + input_img_name))
input_image_mask = cv2.imread(input_mask_path + 'mask_' + input_img_name, cv2.IMREAD_GRAYSCALE)

def main():
    ############################### Step 1 surface normal ########################################
    print("1. Calculating surface normal...")
    # (surface normal method 1)
    sobel_normal = calculate_normal_sobel(input_image)
    
    # (surface normal method 2)
    # step 1.1 calculate depth map
    # - get the depth map
    depth = calculate_depth_face(input_image, img_szie)
    # depth = calculate_depth_relative(input_image) # relative depth estimation
    depth = resize(depth, (img_szie, img_szie))
    depth = np.multiply(depth, 255)
    depth_output = Image.fromarray(depth)
    if depth_output.mode != 'RGB':
        depth_output = depth_output.convert('RGB')
    depth_output.save(output_image_dir + 'depth_' + input_img_name )
    # - derive normal map from depth map
    surface_normal = depth_to_normal(input_image, depth, img_szie)
    normal_output = Image.fromarray(np.uint8(surface_normal))
    if normal_output.mode != 'RGB':
        normal_output = normal_output.convert('RGB')
    normal_output.save(output_image_dir + 'normal_' + input_img_name )
    
    # (surface normal method 3)
    # step 1.1 calculate normal map 
    # calculate_normal(input_image_path, (input_image_path + 'input1.jpg'), output_image_dir)
    print("1. Done surface normal!")
    
    ############################### Step 2 albedo ########################################
    # step 3 calculate albedo map
    print("(TODO)2. Calculating albedo normal...")
    albedo = calculate_albedo(input_image) # TODO
    albedo_ouput = Image.fromarray(np.uint8(albedo))
    if albedo_ouput.mode != 'RGB':
        albedo_ouput = light_ouput.convert('RGB')
    albedo_ouput.save(output_image_dir + 'albedo_' + input_img_name )
    print("(TODO)2. Done albedo!")
    
    ############################### Step 3 HDR to lightmap ########################################
    print("(TODO(PARTIAL))2. Proccessing HDR to lightmap...")
    # step 3 handle HDR
    px = np.float32(Image.open(cube_map_path + 'px.png'))
    nx = np.float32(Image.open(cube_map_path + 'nx.png'))
    py = np.float32(Image.open(cube_map_path + 'py.png'))
    ny = np.float32(Image.open(cube_map_path + 'ny.png'))
    pz = np.float32(Image.open(cube_map_path + 'pz.png'))
    nz = np.float32(Image.open(cube_map_path + 'nz.png'))
    cube_maps = [px, nx, py, ny, pz, nz]
    filtered_cube_maps = prefilter_cube_map(cube_maps) # TODO
    i = 0
    for filtered in filtered_cube_maps:
        filtered_output = Image.fromarray(np.uint8(filtered))
        if filtered_output.mode != 'RGB':
            filtered_output = filtered_output.convert('RGB')
        filtered_output.save(cube_out_path + f"{cubemap_names[i]}.jpg")
        i = i + 1
        
    print("(TODO(PARTIAL))2. Done HDR to lightmap!")
    
    # step 4 shading & compositing
    light = calculate_diffuse(filtered_cube_maps, surface_normal, img_szie)
    light = ndimage.gaussian_filter(light, sigma=(50,50,0)) #TODO replace with our implemetation
    light_ouput = Image.fromarray(np.uint8(light))
    if light_ouput.mode != 'RGB':
        light_ouput = light_ouput.convert('RGB')
    light_ouput.save(output_image_dir + 'light_' + input_img_name )
    
    # step 5 mask normal map & input image
    print("(TODO)5. Masking out backgroud...")
    # auto-matting method need to be implemented
    # TODO
    # light = light * input_image_mask
    # albedo = albedo * input_image_mask
    print("(TODO)5. Done git rid of backgroud!")
    
    # step 6 compositing
    print("6. Compositing final result...")
    final = light / 255 * albedo / 255 * 1.2
    final= final * 255
    final_ouput = Image.fromarray(np.uint8(final))
    if final_ouput.mode != 'RGB':
        final_ouput = light_ouput.convert('RGB')
    final_ouput.save(output_image_dir + 'final_' + input_img_name )
    print("6. Done final result!")

if __name__ == '__main__':
    main()

