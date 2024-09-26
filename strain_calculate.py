import os
import numpy as np
from scipy.io import loadmat
import sys
import matplotlib.pyplot as plt
import matplotlib
sys.path.append("./configs")
sys.path.append("./layers")
sys.path.append("./utils")
from config import config
from utils import zero_to_nan, sub_matrix
from Strain_f_Disp_Subset import Strain_from_Displacement_Subset

def strain_plot(
        exx, eyy, exy,
        exx_min=0, exx_max=1, eyy_min=0, eyy_max=1, exy_min=0, exy_max=1, 
        string='',layout = [3,1], WH=[5,4], 
        save_dir=None, filename=None
        ):
    
    exx = zero_to_nan(exx); eyy = zero_to_nan(eyy); exy = zero_to_nan(exy)
    plt.figure(figsize=(WH[0]*layout[1], WH[1]*layout[0]), dpi=200)
    
    normexx = matplotlib.colors.Normalize(vmin=exx_min, vmax=exx_max)
    normeyy = matplotlib.colors.Normalize(vmin=eyy_min, vmax=eyy_max)
    normexy = matplotlib.colors.Normalize(vmin=exy_min, vmax=exy_max)
    
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(exx, cmap='jet', interpolation='nearest', norm=normexx)
    plt.colorbar()
    plt.axis('off')
    plt.title("Exx"+string, fontsize=10)
    
    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(eyy, cmap='jet', interpolation='nearest', norm=normeyy)
    plt.colorbar()
    plt.axis('off')
    plt.title("Eyy"+string, fontsize=10)
    
    plt.subplot(layout[0], layout[1], 3)
    plt.imshow(exy, cmap='jet', interpolation='nearest', norm=normexy)
    plt.colorbar()
    plt.axis('off')
    plt.title("Exy"+string, fontsize=10)
    
    # Save the figure if save directory and filename are provided
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Strain Figure saved to {file_path}")
    plt.show()


if __name__ == '__main__':
    directory = config['data_path'] +'to_matlab'
    mat_filename = 'result.mat'
    file_path = os.path.join(directory, mat_filename)
    
    if os.path.exists(file_path):
        # Load the .mat file
        data = loadmat(file_path)
        print("File loaded successfully.")
        temp = data['uv'][0,0,:,:]
        roi = np.zeros_like(temp); roi[temp != 0] = 1;
        
        '''
        Compute_list is the selection of the strain field for 
        calculating the deformation diagram of the selected sheet
        '''
        Compute_list = [1, 5]
        
        # Access variables within the .mat file
        # Example: Assuming the .mat file contains a variable named 'uv'
        if 'uv' in data:
            variable_data = data['uv']
            N, C, H, W = variable_data.shape
            for index in Compute_list:
                i = index - 1
                u = variable_data[i,0,:,:]; v = variable_data[i,1,:,:]
                
                Exx, Eyy, Exy = Strain_from_Displacement_Subset(u, v, roi, 1, 3)
                
                exx_non_zero = Exx[roi != 0]; exx_non_zero = exx_non_zero[~np.isnan(exx_non_zero)]
                eyy_non_zero = Eyy[roi != 0]; eyy_non_zero = eyy_non_zero[~np.isnan(eyy_non_zero)]
                exy_non_zero = Exy[roi != 0]; exy_non_zero = exy_non_zero[~np.isnan(exy_non_zero)]
                # Find the minimum and maximum among the non-zero elements
                Exxmin = np.min(exx_non_zero) if exx_non_zero.size > 0 else None
                Exxmax = np.max(exx_non_zero) if exx_non_zero.size > 0 else None
                
                Eyymin = np.min(eyy_non_zero) if eyy_non_zero.size > 0 else None
                Eyymax = np.max(eyy_non_zero) if eyy_non_zero.size > 0 else None
                
                Exymin = np.min(exy_non_zero) if exy_non_zero.size > 0 else None
                Exymax = np.max(exy_non_zero) if exy_non_zero.size > 0 else None
                
                strain_plot(
                        Exx, Eyy, Exy,
                        Exxmin, Exxmax, Eyymin, Eyymax, Exymin, Exymax,
                        string='', layout = [3,1], WH=[5,4], 
                        save_dir=directory, filename=f'example{i+1:04d}_plot.png'
                        )
            
        else:
            print("The variable 'uv' does not exist in the .mat file.")
    else:
        print(f"The file {file_path} does not exist.")
        print("Please run the train.py file firstly, then run the plot_fig.py file")