import os
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append("./configs")
sys.path.append("./layers")
sys.path.append("./utils")
from config import config
from result_plot import result_plot, contourf_plot, error_plot

if __name__ == '__main__':
    directory = config['data_path'] +'to_matlab'
    mat_filename = 'result.mat'
    file_path = os.path.join(directory, mat_filename)
    
    if os.path.exists(file_path):
        # Load the .mat file
        data = loadmat(file_path)
        print("File loaded successfully.")
        
        # Access variables within the .mat file
        # Example: Assuming the .mat file contains a variable named 'uv'
        if 'uv' in data:
            variable_data = data['uv']
            N, C, H, W = variable_data.shape
            for i in range(N):
                u = variable_data[i,0,:,:]; v = variable_data[i,1,:,:]
                u_non_zero = u[u != 0]  # Extract non-zero elements from u
                v_non_zero = v[v != 0]  # Extract non-zero elements from v
                # Find the minimum and maximum among the non-zero elements
                umin = np.min(u_non_zero) if u_non_zero.size > 0 else None
                umax = np.max(u_non_zero) if u_non_zero.size > 0 else None
                vmin = np.min(v_non_zero) if v_non_zero.size > 0 else None
                vmax = np.max(v_non_zero) if v_non_zero.size > 0 else None
                
                result_plot(
                    u, v, u_min=umin, u_max=umax, v_min=vmin, v_max=vmax,
                    string='',layout = [1,2], WH=[5,4], 
                    save_dir=directory, filename=f'example{i+1:04d}_plot.png')
                
            
        else:
            print("The variable 'uv' does not exist in the .mat file.")
    else:
        print(f"The file {file_path} does not exist.")
        print("Please run the train.py file firstly, then run the plot_fig.py file")
    