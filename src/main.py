import os
from surfaceNormal import calculate_normal
dirname = os.path.dirname(os.path.abspath(__file__))
input_image_path = dirname + '/inputs/'
input_mask_path = dirname + '/inputs/'

output_image_dir= dirname + '/outputs/'


def main():
    # step 1.1 calculate normal map
    print("1.1 Calculating surface normal...")
    calculate_normal(input_image_path, (input_image_path + 'input1.jpg'), output_image_dir)
    # step 1.2 mask normal map & input image
    
    # step 2 calculate albedo map
    
    # step 3 handle HDR
    
    # step 4 shading & compositing


if __name__ == '__main__':
    main()

