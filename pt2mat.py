import torch
import os, sys, glob
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

#example command python pt2mat.py models models_mat
pt_dir = sys.argv[1]  # dir of torch models (pt_dir/model_name/ckpt_{30,60,90}.pth)
mat_dir = sys.argv[2] # dir to store mat files (mat_dir/model_name/ckpt_90.mat)
for pt_file in tqdm(glob.glob('{}/*/ckpt_90.pth'.format(pt_dir))):
    org_m = torch.load(pt_file)
    new_m = {}
    for k,v in org_m.items():
        #print(k, v.shape)
        new_v = v.cpu().numpy()
        new_m[k] = new_v

    # note that the valid layers are hidden1 hidden2 predict
    mat_dir2 = os.path.dirname(pt_file.replace(pt_dir, mat_dir))
    if not os.path.exists(mat_dir2):
        os.makedirs(mat_dir2)
    mat_file = os.path.join(mat_dir2, 'ckpt_90.mat')
    savemat(mat_file, new_m)
