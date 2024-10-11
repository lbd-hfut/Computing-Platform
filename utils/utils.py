# src/utils.py
import numpy as np
import torch
import os
from PIL import Image
import scipy

def model_data_collect(ref_image_path, def_image_path, roi_path):
    RG, DG, roi = load_images(ref_image_path, def_image_path, roi_path)
    H,L = RG.shape; 
    y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
    IX,IY = np.meshgrid(x, y)
    XY_roi = np.column_stack(np.where(roi == 1))
    return RG, DG, roi, IX, IY, XY_roi
    
def data_collect(RG, roi):
    H,L = RG.shape; 
    y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
    IX,IY = np.meshgrid(x, y)
    XY_roi = np.column_stack(np.where(roi == 1))
    return IX, IY, XY_roi

def load_images(ref_image_path, def_image_path, roi_path):
    ref_image = Image.open(ref_image_path)
    ref_gray = ref_image.convert('L')
    ref_gray = np.array(ref_gray)
    
    def_image = Image.open(def_image_path)
    def_gray = def_image.convert('L')
    def_gray = np.array(def_gray)
    
    roi_image = Image.open(roi_path)
    roi_gray = roi_image.convert('L')
    roi_gray = np.array(roi_gray)
    min_val = np.min(roi_gray)
    max_val = np.max(roi_gray)
    roi = (roi_gray - min_val) / (max_val - min_val)
    return ref_gray, def_gray, roi

def create_meshgrid(x_size, y_size):
    x_list = np.linspace(-1, 1, x_size)
    y_list = np.linspace(-1, 1, y_size)
    IX, IY = np.meshgrid(x_list, y_list)
    return IX, IY

def save_mat(model, IX, IY, string1=''):
    u1,v1 = model.predict(IX, IY)
    data_to_save = {'v': v1, 'u':'u1'}
    string = f'../model_parameter'
    scipy.io.savemat(string + string1, data_to_save)
    
def zero_to_nan(matrix):
    matrix = np.array(matrix) 
    non_zero_row_indices = np.where(np.any(matrix != 0, axis=1))[0]
    non_zero_col_indices = np.where(np.any(matrix != 0, axis=0))[0]
    submatrix = matrix[non_zero_row_indices[:, None], non_zero_col_indices]
    submatrix[submatrix == 0] = np.nan
    return submatrix 

def sub_matrix(matrix):
    matrix = np.array(matrix) 
    non_zero_row_indices = np.where(np.any(matrix != 0, axis=1))[0]
    non_zero_col_indices = np.where(np.any(matrix != 0, axis=0))[0]
    submatrix = matrix[non_zero_row_indices[:, None], non_zero_col_indices]
    return submatrix

def save_checkpoint(model, epoch, loss, path_checkpoint):
    checkpoint = {
        'modelu_state_dict': model[0].state_dict(),
        'modelv_state_dict': model[1].state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, path_checkpoint)
