import cv2
import torch
import numpy as np
import math
from vendors.depth_predict import face_depth

# get relative depth map
def calculate_depth_relative(img):
    # load the pretrained model
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    
    # transform img
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform

    input_normalized = transform(img).to(device)
    
    with torch.no_grad():
        prediction = model(input_normalized)
        # resize back
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()
    return depth

# a precise depth map for face
def calculate_depth_face(img, size):
    model = face_depth.Model()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load('./vendors/depth_predict/weights_model.pth')) # download model https://nuigalwayie-my.sharepoint.com/:u:/g/personal/f_khan4_nuigalway_ie/EepkuVajAhdIjZoQm5Weyx4BjXcEZy-uw5OWxxMXq1WJPA?e=rv3aSY
    model.eval()

    input_img = img / 255.

    t1 = input_img.transpose(-1, 0, 1).reshape(1, 3, size, size)
    t2 = torch.from_numpy(t1).float()
    t3 = model(t2.cpu())
    t4 = t3.detach().cpu().numpy()
    depth = t4/25.
    depth = np.squeeze(depth)
    depth = depth.astype(np.uint8)
    # return range from 0-1
    return depth

# derive normal map from depth map
def depth_to_normal(img, depth, size):
    # TODO (Yimin) interpolate depth
    normal = np.full_like(img, 0., dtype=float)
    for y in range(size):
        for x in range(size):
            deltaX = 0.
            deltaY = 0.
            if (x + 3) < size:
                deltaX = (depth[y][x + 3] - depth[y][x - 3])

            if (y + 3) < size:
                deltaY = (depth[y + 3][x] - depth[y - 3][x])
            
            n_dir = (-deltaX, -deltaY, 1.)
            # normalized gradient
            gradient = math.sqrt(n_dir[0]**2. + n_dir[1]**2. + n_dir[2]**2.)
            # remap & normalize normal and map to 0-255
            normal[y, x] = [(-deltaX / gradient / 2. + 0.5) * 255., (-deltaY / gradient / 2. + 0.5) * 255., (n_dir[2] / gradient / 2. + 0.5) * 255.]

    return normal
