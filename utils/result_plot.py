import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from utils import zero_to_nan, sub_matrix
import scipy.io as io

def result_plot(u, v, u_min=0, u_max=1, v_min=0, v_max=1,string='',layout = [1,2], WH=[5,4], save_dir=None, filename=None):
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
    # Save the figure if save directory and filename are provided
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing
    
def contourf_plot(u, v, N, u_min=0, u_max=1, v_min=0, v_max=1,string='',layout = [1,2], WH=[4,4], save_dir=None, filename=None):
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
    # Save the figure if save directory and filename are provided
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing
    
def error_plot(u_error, v_error, u_min=1, u_max=0, v_min=0, v_max=1, string='',layout = [1,2], WH=[4,4], save_dir=None, filename=None):
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
    # Save the figure if save directory and filename are provided
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing
    
def to_matlab(path, filename, uv):
    if not os.path.exists(path+'to_matlab'):
        os.mkdir(path+'to_matlab')
    io.savemat(path+'to_matlab/'+filename+'.mat',{'uv':uv})
    
def to_txt(path, filename, xyuv):
    if not os.path.exists(path+'to_matlab'):
        os.mkdir(path+'to_matlab')
    output_file_path = path+'to_matlab/'+filename+'.txt'
    
    N, _, _ = xyuv.shape
    with open(output_file_path, 'w') as f:
        f.write("The first and second columns represent the pixel coordinates x, y. "
                "The third and fourth columns represent the displacement values u, v\n\n")
        f.write("\n")
        for i in range(N):
            f.write(f"this is the {i+1}-th solution\n")
            np.savetxt(f, xyuv[i], fmt='%.5f')
            f.write("\n")