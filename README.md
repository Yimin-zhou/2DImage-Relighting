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
 
- Implementation(roughly 4-5 steps):
  - [] Get depth map, then derive normal map from the depth https://pytorch.org/hub/intelisl_midas_v2/
    (Alternative)- [X] calculate normal map, based on PIFuHD: Multi-Level Pixel-Aligned Implicit Function for
High-Resolution 3D Human Digitization∗
 https://arxiv.org/pdf/2004.00452.pdf (some code and pre-trained models provided by the paper are used in our project)
 
  - [ ] mask out unwanted parts (maybe add auto-extract function)
  
  - [ ] calculate albedo map

  - [ ] handle HDR

  - [ ] shading & compositing

 - (Reminder: might need to change the order of some steps & mask normal map)
