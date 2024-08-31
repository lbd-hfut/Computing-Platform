import torch
import torch.nn.functional as F
import numpy as np
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def criterion_warmup(UV, XY_roi, XY, Iref, Idef, ROI, scale):
    # Adjust the shape of the vector to match the shape of the image
    coords = XY_roi
    U = torch.zeros_like(Iref).to(device)
    V = torch.zeros_like(Iref).to(device)
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    U[y_coords, x_coords] = UV[:, 0] * scale[0]
    V[y_coords, x_coords] = UV[:, 1] * scale[1]
    
    # Interpolate a new deformed image
    target_height = Idef.shape[0]; target_width = Idef.shape[1]
    u = -U/(target_width/2); v = -V/(target_height/2)
    uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
    X_new = XY + uv_displacement
    new_Idef = F.grid_sample(Iref.view(1, 1, target_height, target_width), 
                                X_new.view(1, target_height, target_width, 2), 
                                mode='bilinear', align_corners=True)
    
    # calculate the loss
    abs_error = (new_Idef[0, 0] - Idef)**2 * ROI
    mse = torch.sum(abs_error)/XY_roi.shape[0]
    absolute_error = torch.abs(new_Idef[0, 0] - Idef) * ROI
    mae = torch.sum(absolute_error)/XY_roi.shape[0]
    return mse, mae


def criterion_train(UV, XY_roi, XY, Iref, Idef, ROI, scale):
    # Adjust the shape of the vector to match the shape of the image
    coords = XY_roi
    U = torch.zeros_like(Iref).to(device)
    V = torch.zeros_like(Iref).to(device)
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    U[y_coords, x_coords] = UV[:, 0] * scale[0]
    V[y_coords, x_coords] = UV[:, 1] * scale[1]
    
    # Interpolate a new deformed image
    target_height = Idef.shape[0]; target_width = Idef.shape[1]
    u = -U/(target_width/2); v = -V/(target_height/2)
    uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
    X_new = XY + uv_displacement
    new_Idef = F.grid_sample(Iref.view(1, 1, target_height, target_width), 
                                X_new.view(1, target_height, target_width, 2), 
                                mode='bilinear', align_corners=True)
    
    # calculate the loss
    abs_error = (new_Idef[0, 0] - Idef)**2 * ROI
    abs_error = torch.log10(1+abs_error)
    mse_lg = torch.sum(abs_error)/XY_roi.shape[0]
    absolute_error = torch.abs(new_Idef[0, 0] - Idef) * ROI
    mae = torch.sum(absolute_error)/XY_roi.shape[0]
    return mse_lg, mae