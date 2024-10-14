import os
import torch
import numpy as np
import sys
import math
import scipy.io as io
import matplotlib  
matplotlib.use('Agg')

sys.path.append("./layers")
sys.path.append("./utils")

from scale_select import sift_matching_within_roi, remove_outliers, match_plot
from configs.config import config

if __name__ == '__main__':
    max_matches=3000 # 
    safety_factor = 1.2
    '''
    The threshold for standard deviation multiples, 
    points exceeding this threshold are considered outliers and default to 3
    '''
    threshold = 3
    
    device = torch.device('cpu')
    
    path = config['data_path']
    if not os.path.exists(path+'scale_information'):
        os.mkdir(path+'scale_information')
    directory = path +'scale_information'
    
    image_files = np.array([x.path for x in os.scandir(config['data_path'])
                         if (x.name.endswith(".bmp") or
                         x.name.endswith(".png") or 
                         x.name.endswith(".JPG"))])
    image_files.sort()
    
    rfimage_files = [image_files[0]]
    mask_files = [image_files[-1]]
    dfimage_files = image_files[1:-1]
    
    batchframes = config['Batchframes']
    N = len(dfimage_files)
    numbers = list(range(N))
    BatchList = [numbers[i:i+batchframes] for i in range(0, len(numbers), batchframes)]
    BATCH = math.floor(N / batchframes)
    
    
    SCALE_LIST = []
    for batch, iLIst in enumerate(BatchList):
        if batch == BATCH:
            i = iLIst[-1]
            data = sift_matching_within_roi(
                rfimage_files[0], dfimage_files[i], mask_files[0], max_matches
                )
            data = remove_outliers(data, threshold)
            match_plot(
                data, rfimage_files[0], dfimage_files[i], 
                save_dir=directory, filename=f'example{i+1:03d}_match.png'
                )
            displacements = np.abs(data[:,4:6])
            u_max = np.max(data[:,4])
            v_max = np.max(data[:,5])
            u_min = np.min(data[:,4])
            v_min = np.min(data[:,5])
            u_scale = ((u_max - u_min)/2) * safety_factor
            v_scale = ((v_max - v_min)/2) * safety_factor
            u_scale = 1 if round(u_scale) == 0 else u_scale
            v_scale = 1 if round(v_scale) == 0 else v_scale
            # u_mean = int(np.mean(displacements[:,0]))/u_scale
            # v_mean = int(np.mean(displacements[:,1]))/v_scale
            u_mean = int(np.mean(data[:,4:5]))
            v_mean = int(np.mean(data[:,5:6]))
            for i in range(len(iLIst)):
                SCALE_LIST.append([int(u_scale), int(v_scale), u_mean, v_mean])
        else:
            i = iLIst[batchframes//2]
            data = sift_matching_within_roi(
                rfimage_files[0], dfimage_files[i], mask_files[0], max_matches
                )
            data = remove_outliers(data, threshold)
            match_plot(
                data, rfimage_files[0], dfimage_files[i], 
                save_dir=directory, filename=f'example{i+1:03d}_match.png'
                )
            displacements = np.abs(data[:,4:6])
            u_max = np.max(data[:,4])
            v_max = np.max(data[:,5])
            u_min = np.min(data[:,4])
            v_min = np.min(data[:,5])
            u_scale = ((u_max - u_min)/2) * safety_factor
            v_scale = ((v_max - v_min)/2) * safety_factor
            u_scale = 1 if round(u_scale) == 0 else u_scale 
            v_scale = 1 if round(v_scale) == 0 else v_scale
            u_mean = round(np.mean(data[:,4:5]))
            v_mean = round(np.mean(data[:,5:6]))
            for i in range(len(iLIst)):
                SCALE_LIST.append([int(u_scale), int(v_scale), u_mean, v_mean])
    
    io.savemat(directory+'/SCALE.mat',{'scale':SCALE_LIST})
    print("The scale list is saved to "+directory+'/SCALE.mat')
