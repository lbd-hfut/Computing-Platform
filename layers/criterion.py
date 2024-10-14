import torch
import torch.nn.functional as F
import numpy as np
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def criterion_warmup(UV, XY_roi, XY, Iref, Idef, ROI, scale):
    # Adjust the shape of the vector to match the shape of the image
    U = torch.zeros_like(Iref)
    V = torch.zeros_like(Iref)
    y_coords, x_coords = XY_roi[:, 0], XY_roi[:, 1]
    U[y_coords, x_coords] = UV[:, 0] * scale[0] + scale[2] 
    V[y_coords, x_coords] = UV[:, 1] * scale[1] + scale[3] 
    
    # Interpolate a new deformed image
    target_height = Idef.shape[0]; target_width = Idef.shape[1]
    u = -U/(target_width/2); v = -V/(target_height/2)
    uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
    X_new = XY + uv_displacement
    # print(f"Iref: {Iref.shape}; Idef: {X_new.shape}")
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
    U = torch.zeros_like(Iref)
    V = torch.zeros_like(Iref)
    y_coords, x_coords = XY_roi[:, 0], XY_roi[:, 1]
    U[y_coords, x_coords] = UV[:, 0] * scale[0] + scale[2]
    V[y_coords, x_coords] = UV[:, 1] * scale[1] + scale[3]
    
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

def criterion_warmup_lgd(UV, XY_roi, XY, Iref, Idef, ROI, scale):
    # Adjust the shape of the vector to match the shape of the image
    U = torch.zeros_like(Iref)
    V = torch.zeros_like(Iref)
    y_coords, x_coords = XY_roi[:, 0], XY_roi[:, 1]
    U[y_coords, x_coords] = UV[:, 0] * scale[0] + scale[2] 
    V[y_coords, x_coords] = UV[:, 1] * scale[1] + scale[3] 
    
    # Interpolate a new deformed image
    target_height = Idef.shape[0]; target_width = Idef.shape[1]
    u = -U/(target_width/2); v = -V/(target_height/2)
    uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
    X_new = XY + uv_displacement
    new_Idef = F.grid_sample(Iref.view(1, 1, target_height, target_width), 
                                X_new.view(1, target_height, target_width, 2), 
                                mode='bilinear', align_corners=True)
    new_ROI = F.grid_sample(Iref.view(1, 1, target_height, target_width), 
                                X_new.view(1, target_height, target_width, 2), 
                                mode='bilinear', align_corners=False)
    new_ROI = new_ROI > 0
    # calculate the loss
    abs_error = (new_Idef[0, 0] - Idef)**2 * new_ROI
    mse = torch.sum(abs_error)/XY_roi.shape[0]
    absolute_error = torch.abs(new_Idef[0, 0] - Idef) * ROI
    mae = torch.sum(absolute_error)/XY_roi.shape[0]
    return mse, mae


def criterion_train_lgd(UV, XY_roi, XY, Iref, Idef, ROI, scale):
    # Adjust the shape of the vector to match the shape of the image
    U = torch.zeros_like(Iref)
    V = torch.zeros_like(Iref)
    y_coords, x_coords = XY_roi[:, 0], XY_roi[:, 1]
    U[y_coords, x_coords] = UV[:, 0] * scale[0] + scale[2] 
    V[y_coords, x_coords] = UV[:, 1] * scale[1] + scale[3] 
    
    # Interpolate a new deformed image
    target_height = Idef.shape[0]; target_width = Idef.shape[1]
    u = -U/(target_width/2); v = -V/(target_height/2)
    uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
    X_new = XY + uv_displacement
    new_Idef = F.grid_sample(Iref.view(1, 1, target_height, target_width), 
                                X_new.view(1, target_height, target_width, 2), 
                                mode='bilinear', align_corners=True)
    new_ROI = F.grid_sample(Iref.view(1, 1, target_height, target_width), 
                                X_new.view(1, target_height, target_width, 2), 
                                mode='bilinear', align_corners=False)
    new_ROI = new_ROI > 0
    # calculate the loss
    abs_error = (new_Idef[0, 0] - Idef)**2 * new_ROI
    abs_error = torch.log10(1+abs_error)
    mse_lg = torch.sum(abs_error)/XY_roi.shape[0]
    absolute_error = torch.abs(new_Idef[0, 0] - Idef) * ROI
    mae = torch.sum(absolute_error)/XY_roi.shape[0]
    return mse_lg, mae

def Straincompatibility(UV, XY_roi, ROI):
    target_height = ROI.shape[0]
    target_width = ROI.shape[1]
    
    u = torch.zeros_like(ROI)
    v = torch.zeros_like(ROI)
    y_coords, x_coords = XY_roi[:, 0], XY_roi[:, 1]
    u[y_coords, x_coords] = UV[:, 0]
    v[y_coords, x_coords] = UV[:, 1]

    # 计算 ex, ey, exy
    ex = torch.diff(u, dim=1)[1:target_height+1,:]   # (h,w)->(h,w-1)->(h-1,w-1)
    ey = torch.diff(v, dim=0)[:,1:target_width+1]    # (h,w)->(h-1,w)->(h-1,w-1)
    exy = 0.5 * (ex + ey)                            # (h-1,w-1)
    # 计算 ex1, ex2, ey1, ey2, exy1, exy2
    ex1 = torch.diff(ex, dim=1)                      # (h-1,w-1)->(h-1,w-2)
    ex2 = torch.diff(ex1, dim=1)[2:target_height,:]  # (h-1,w-2)->(h-1,w-3)->(h-3,w-3)

    ey1 = torch.diff(ey, dim=0)                      # (h-1,w-1)->(h-2,w-1)
    ey2 = torch.diff(ey1, dim=0)[:,2:target_width]   # (h-2,w-1)->(h-3,w-1)->(h-3,w-3)

    exy1 = torch.diff(exy, dim=1)[1:target_height,:] # (h-1,w-1)->(h-1,w-2)->(h-2,w-2)
    exy2 = torch.diff(exy1, dim=0)[:,1:target_width] # (h-2,w-2)->(h-3,w-2)->(h-3,w-3)
    result = torch.sum((ex2 + ey2 - exy2) ** 2)
    return result