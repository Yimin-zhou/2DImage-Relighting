# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
from .recon import reconWrapper
import argparse


###############################################################################################
##                   Setting
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, default='/mnt/d/Programming/Assignments/Relighting/src/inputs')
parser.add_argument('-o', '--out_path', type=str, default='/mnt/d/Programming/Assignments/Relighting/src/outputs')
parser.add_argument('-c', '--ckpt_path', type=str, default='./checkpoints/pifuhd.pt')
parser.add_argument('-r', '--resolution', type=int, default=1024)
parser.add_argument('--use_rect', action='store_true', help='use rectangle for cropping')
args = parser.parse_args()
###############################################################################################
##                   Upper PIFu
###############################################################################################

resolution = str(args.resolution)

start_id = -1
end_id = -1
print(args.ckpt_path)
cmd = ['--dataroot', args.input_path, '--results_path', args.out_path,\
       '--loadSize', '1024', '--resolution', '1024', '--load_netMR_checkpoint_path', \
       args.ckpt_path,\
       '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
reconWrapper(cmd, args.use_rect)

