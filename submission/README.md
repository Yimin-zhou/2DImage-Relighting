### How to start
 1. Install `Anaconda` on your machine.
 2. In root directory run 
    `conda env create -f environment.yml` & `conda activate relighting`
 3. Download pre-trained NN model
    - For precise face depth estimation `depth_normal.py (def calculate_depth_face)` : Download: https://nuigalwayie-my.sharepoint.com/:u:/g/personal/f_khan4_nuigalway_ie/EepkuVajAhdIjZoQm5Weyx4BjXcEZy-uw5OWxxMXq1WJPA?e=rv3aSY to `src/vendors/depth_predict`
    - For monocular depth estimation `depth_normal.py (def calculate_depth_relative)` : Automatic Downloading
    - For full body surface normal `full_body_normal.py`: Redirect to `src/venders/` run `sh ./scripts/download_trained_model.sh` to download the pretrained model
 4. Redirect to `src/` run `python3 main.py`
 5. You will see output images inside folder `outputs`

 ### Pipeline
 - The Pipeline: `Surface Normal` -> `Albedo` -> `Cubmap filering` -> `Masking` -> `Shading`
  1.1 (surfaceNormal method 1) use Sobel filter to get normal map(`sobel_normal.py`).
  1.2 (surfaceNormal method 2) derive normal map from the depth(`depth_normal.py`).
  1.3 (surfaceNormal method 3) calculate normal map, based on full-body image(`depth_normal.py`).
 
  2. albedo map(`albedo.py`)
  
  3. prefilter cubemap(`hdr_filtering.py`)

  4. light map(`shading.py`)
