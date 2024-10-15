# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:58:30 2024

@author: 28622
"""
import torch
import numpy as np
from utils import zero_to_nan, sub_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import matplotlib

def debug_plot(model, Ixy, XY_roi, ROI, SCALE, i):
    model[0].eval()
    model[1].eval()
    with torch.no_grad():
        U = model[0](Ixy); V = model[1](Ixy)
    UV = torch.cat((U, V), dim=1)
    UV[:, 0] = UV[:, 0] * SCALE['scale'][i][0] + SCALE['scale'][i][2]
    UV[:, 1] = UV[:, 1] * SCALE['scale'][i][1] + SCALE['scale'][i][3]
    
    coords = XY_roi
    U = torch.zeros_like(ROI).to(device)
    V = torch.zeros_like(ROI).to(device)
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    U[y_coords, x_coords] = UV[:, 0]
    V[y_coords, x_coords] = UV[:, 1]
    
    U = U.cpu().detach().numpy()
    V = V.cpu().detach().numpy()
    
    u_non_zero = U[U != 0]  # Extract non-zero elements from u
    v_non_zero = V[V != 0]  # Extract non-zero elements from v

    # Find the minimum and maximum among the non-zero elements
    umin = np.min(u_non_zero) if u_non_zero.size > 0 else None
    umax = np.max(u_non_zero) if u_non_zero.size > 0 else None
    vmin = np.min(v_non_zero) if v_non_zero.size > 0 else None
    vmax = np.max(v_non_zero) if v_non_zero.size > 0 else None
    
    u = zero_to_nan(U); v = zero_to_nan(V)
    plt.figure(figsize=(16, 6), dpi=200)
    normu = matplotlib.colors.Normalize(vmin=umin, vmax=umax)
    normv = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 1)
    plt.imshow(u, cmap='jet', interpolation='nearest', norm=normu)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(v, cmap='jet', interpolation='nearest', norm=normv)
    plt.colorbar()
    plt.axis('off')
    file_path = './logs/for_test.png'
    plt.savefig(file_path, bbox_inches='tight')
    print(f"Figure saved to {file_path}")
    plt.close()
    