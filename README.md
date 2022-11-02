# Image-Relighting

### How to start
 1. Install `Anaconda` on your machine.
 2. In root directory run 
    `conda env create -f environment.yml` & `conda activate relighting`
 3. Redirect to `src/venders/` run `sh ./scripts/download_trained_model.sh` to download the pretrained model
 4. Redirect to `src/` run `python3 main.py`
 5. You will see output images inside folder `outputs`
 (Warning: this only works for full-body(human) photos)
 
 ### About this project
 - The pipline is inspired by Total Relighting: Learning to Relight Portraits for Background Replacement https://augmentedperception.github.io/total_relighting/total_relighting_paper.pdf (Google, 2021)                                                                         
 The implementation steps are different from the paper.

- The Pipeline: `Surface Normal` -> `Image Matting` -> `Albedo` -> `HDR filering` -> `Shading`
  - [X] (surfaceNormal method 1) Get normal map directly (sobel). http://citebay.com/how-to-cite/sobel-filter/
  - [X] (surfaceNormal method 2) Get face/relative depth map, then derive normal map from the depth https://pytorch.org/hub/intelisl_midas_v2/ & download model https://nuigalwayie-my.sharepoint.com/:u:/g/personal/f_khan4_nuigalway_ie/EepkuVajAhdIjZoQm5Weyx4BjXcEZy-uw5OWxxMXq1WJPA?e=rv3aSY & https://github.com/khan9048/Facial_depth_estimation & https://courses.cs.washington.edu/courses/cse590b/02au/hdrc.pdf
  - [X] (surfaceNormal method 3) calculate normal map, based on PIFuHD: Multi-Level Pixel-Aligned Implicit Function for
High-Resolution 3D Human Digitization∗
 https://arxiv.org/pdf/2004.00452.pdf (some code and pre-trained models provided by the paper are used in our project)
 
  - [ ] mask out unwanted parts (maybe add auto-extract function)
  
  - [ ] calculate albedo map

  - [ ] handle HDR, TODO transform HDR to cube map. (Real Time Readering, 6.2.4)

  - [ ] shading & compositing

 - (Reminder: might need to change the order of some steps & mask normal map)
