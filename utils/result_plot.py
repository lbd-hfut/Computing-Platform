import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from utils import zero_to_nan, sub_matrix
import scipy.io as io

def result_plot(u, v, u_min=0, u_max=1, v_min=0, v_max=1,string='',layout = [1,2], WH=[5,4]):
    u = zero_to_nan(u); v = zero_to_nan(v)
    plt.figure(figsize=(WH[0]*layout[1], WH[1]*layout[0]), dpi=200)
    normu = matplotlib.colors.Normalize(vmin=u_min, vmax=u_max)
    normv = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(u, cmap='jet', interpolation='nearest', norm=normu)
    plt.colorbar()
    plt.axis('off')
    plt.title("Train: u predicted"+string, fontsize=10)
    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(v, cmap='jet', interpolation='nearest', norm=normv)
    plt.colorbar()
    plt.axis('off')
    plt.title("Train: v predicted"+string, fontsize=10)
    
def contourf_plot(u, v, N, u_min=0, u_max=1, v_min=0, v_max=1,string='',layout = [1,2], WH=[4,4]):
    u = sub_matrix(u); v = sub_matrix(v);
    u_sub1 = np.flip(u, axis=0)
    v_sub1 = np.flip(v, axis=0)
    H,L = u_sub1.shape; 
    y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
    IX,IY = np.meshgrid(x, y)
    fig, (ax1, ax2) = plt.subplots(layout[0], layout[1], figsize=(WH[0]*layout[1], WH[1]*layout[0]), dpi=200)
    c1 = ax1.contourf(IX, IY, u_sub1, N, cmap='jet')
    plt.colorbar(c1, ax=ax1, orientation='vertical')
    ax1.axis('off'); c1.set_clim(u_min, u_max)
    
    c2 = ax2.contourf(IX, IY, v_sub1, N, cmap='jet')
    plt.colorbar(c2, ax=ax2, orientation='vertical')
    ax2.axis('off'); c2.set_clim(v_min, v_max)
    
def error_plot(u_error, v_error, u_min=1, u_max=0, v_min=0, v_max=1, string='',layout = [1,2], WH=[4,4]):
    u_error = zero_to_nan(u_error); v_error = zero_to_nan(v_error)
    plt.figure(figsize=(WH[0]*layout[1], WH[1]*layout[0]), dpi=200)
    normu = matplotlib.colors.Normalize(umin=u_min, umax=u_max)
    normv = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(u_error, cmap='jet', interpolation='nearest', norm=normu)
    plt.colorbar()
    plt.axis('off')
    plt.title("Error of u"+string, fontsize=10)
    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(v_error, cmap='jet', interpolation='nearest', norm=normv)
    plt.colorbar()
    plt.axis('off')
    plt.title("Error of v"+string, fontsize=10)
    
def to_matlab(path, filename, uv):
    if not os.path.exists(path+'to_matlab'):
        os.mkdir(path+'to_matlab')
    io.savemat(path+'to_matlab/'+filename+'.mat',{'uv':uv})