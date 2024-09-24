import os
import torch
import numpy as np
import sys
import math
import scipy.io as io
sys.path.append("./layers")
sys.path.append("./utils")

from scale_select import sift_matching_within_roi, remove_outliers, match_plot
from configs.config import config

if __name__ == '__main__':
    max_matches=100 # 
    safety_factor = 1.2
    '''
    The threshold for standard deviation multiples, 
    points exceeding this threshold are considered outliers and default to 3
    '''
    threshold = 1
    
    device = torch.device('cpu')
    
    path = config['data_path']
    if not os.path.exists(path+'scale_information'):
        os.mkdir(path+'scale_information')
    directory = path +'scale_information'
    
    rfimage_files = np.array(
            [x.path for x in os.scandir(config['data_path']) 
             if (x.name.endswith(".bmp") or 
             x.name.endswith(".png") or 
             x.name.endswith(".JPG")) and 
             x.name.startswith("r")])
    
    dfimage_files = np.array(
            [x.path for x in os.scandir(config['data_path']) 
            if (x.name.endswith(".bmp") or 
            x.name.endswith(".png") or 
            x.name.endswith(".JPG")) and 
            x.name.startswith("d")])
    
    mask_files = np.array(
            [x.path for x in os.scandir(config['data_path']) 
            if (x.name.endswith(".bmp") or 
            x.name.endswith(".png") or 
            x.name.endswith(".JPG")) and 
            x.name.startswith("mask")])
    
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
            u_max = round(np.max(displacements[:,0])*safety_factor) # 1.5 is the safety factor
            v_max = round(np.max(displacements[:,1])*safety_factor)
            u_max = 1 if u_max == 0 else u_max 
            v_max = 1 if v_max == 0 else v_max
            for i in range(len(iLIst)):
                SCALE_LIST.append([u_max, v_max])
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
            u_max = round(np.max(displacements[:,0])*safety_factor) # 1.5 is the safety factor
            v_max = round(np.max(displacements[:,1])*safety_factor)
            u_max = 1 if u_max == 0 else u_max 
            v_max = 1 if v_max == 0 else v_max
            for i in range(len(iLIst)):
                SCALE_LIST.append([u_max, v_max])
    
    io.savemat(directory+'/SCALE.mat',{'scale':SCALE_LIST})
    print("The scale list is saved to "+directory+'/SCALE.mat')
