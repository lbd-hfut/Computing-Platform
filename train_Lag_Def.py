'''
This is a program for handling large deformations. 
It is recommended to lower the learning rate by 0.00001~0.0001
'''

import os
import torch
import numpy as np
import random
import sys
import math
from scipy.io import loadmat
sys.path.append("./layers")
sys.path.append("./utils")

from criterion import criterion_warmup_lgd, criterion_train_lgd
from FCNN import DNN
from utils import save_checkpoint
from readData import lbdDataset, collate_fn
from result_plot import to_matlab, to_txt

from configs.config import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    '''Calculate the memory size of the model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

if not os.path.exists(config['data_path']+'scale_information/'+'SCALE.mat'):
    raise ValueError("please run scale_list.py firstly")
else:
    SCALE = loadmat(config['data_path']+'scale_information/'+'SCALE.mat')

U_PAT_FAC = 1; V_PAT_FAC = 1
U_MAX_ZONE= SCALE['scale'][-1][0] + abs(SCALE['scale'][-1][2])
V_MAX_ZONE = SCALE['scale'][-1][1] + abs(SCALE['scale'][-1][3])
if U_MAX_ZONE > 5 * V_MAX_ZONE:
    V_PAT_FAC = 3; U_PAT_FAC = 0.8
elif V_MAX_ZONE > 5 * U_MAX_ZONE:
    U_PAT_FAC = 3; V_PAT_FAC = 0.8

seed_everything(1234)
modelu = DNN(config['layers']).to(device)
modelv = DNN(config['layers']).to(device)
model = [modelu, modelv]

# set the optimizer
model[0].get_optimizer(config)
model[1].get_optimizer(config)



def closure1(model, Ixy, XY_roi, XY, Iref, Idef, ROI, scale, i):
    model[0].optimizer_lbfgs.zero_grad()
    model[1].optimizer_lbfgs.zero_grad()
    U = model[0](Ixy); V = model[1](Ixy)
    UV = torch.cat((U, V), dim=1)
    loss, mae = criterion_warmup_lgd(UV, XY_roi, XY, Iref, Idef, ROI, scale)
    loss.backward()
    config['epoch'] += 1
    model[0].Earlystop(mae, model[0], i , config['epoch'])
    model[1].Earlystop(mae, model[1], i , config['epoch'])
    return loss, mae

def closure2(model, Ixy, XY_roi, XY, Iref, Idef, ROI, scale, i):
    model[0].optimizer_lbfgs.zero_grad()
    model[1].optimizer_lbfgs.zero_grad()
    U = model[0](Ixy); V = model[1](Ixy)
    UV = torch.cat((U, V), dim=1)
    loss, mae = criterion_train_lgd(UV, XY_roi, XY, Iref, Idef, ROI, scale)
    loss.backward()
    config['epoch'] += 1
    model[0].Earlystop(mae, model[0], i , config['epoch'])
    model[1].Earlystop(mae, model[1], i , config['epoch'])
    return loss, mae

def warm_up(i, Ixy, XY_roi, XY, RG, DG, ROI):
    if config['warm_adam_epoch'] != 0:
        model[0].Earlystop_set(config['patience_adam']*U_PAT_FAC, config['delta_warm_adam']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_adam']*V_PAT_FAC, config['delta_warm_adam']/V_PAT_FAC)
        model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
        print("warm adam start:")
        for iter in range(config['warm_adam_epoch']):
            model[0].optimizer_adam.zero_grad()
            model[1].optimizer_adam.zero_grad()
            U = model[0](Ixy); V = model[1](Ixy)
            UV = torch.cat((U, V), dim=1)
            loss, mae = criterion_warmup_lgd(
                UV, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i]
                )
            loss.backward()
            model[0].optimizer_adam.step()
            model[1].optimizer_adam.step()
            config['epoch'] += 1
            model[0].Earlystop(mae, model[0], i , config['epoch'])
            model[1].Earlystop(mae, model[1], i , config['epoch'])
            if config['epoch']%config['print_feq'] == 1:
                epoch =  config['epoch'] 
                print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
            if model[0].early_stop or model[1].early_stop:
                if model[0].early_stop:
                    model[0].freeze_all_parameters()
                if model[1].early_stop:
                    model[1].freeze_all_parameters()
                if model[0].early_stop and model[1].early_stop:
                    print("warm adam early stopping")
                    break
    if config['warm_bfgs_epoch'] > config['max_iter']:
        model[0].Earlystop_set(config['patience_lbfgs']*U_PAT_FAC, config['delta_warm_lbfgs']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_lbfgs']*V_PAT_FAC, config['delta_warm_lbfgs']/V_PAT_FAC)
        model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
        print("warm lbfgs start:") 
        for iter in range(config['warm_bfgs_epoch']//config['max_iter']):
            def closure1_wrapper():
                loss, mae = closure1(
                    model, Ixy, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i], i
                    )
                if config['epoch']%config['print_feq'] == 1:
                    epoch =  config['epoch'] 
                    print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                return loss
            loss = closure1_wrapper()
            model[0].optimizer_lbfgs.step(closure1_wrapper)
            model[1].optimizer_lbfgs.step(closure1_wrapper)
            if model[0].early_stop or model[1].early_stop:
                if model[0].early_stop:
                    model[0].freeze_all_parameters()
                if model[1].early_stop:
                    model[1].freeze_all_parameters()
                if model[0].early_stop and model[1].early_stop:
                    print("warm adam early stopping")
                    break
            
def train_stage(i, Ixy, XY_roi, XY, RG, DG, ROI):
    if config['train_adam_epoch'] != 0:
        model[0].optimizer_adam.param_groups[0]['lr'] = config['train_lr']
        model[1].optimizer_adam.param_groups[0]['lr'] = config['train_lr']
        model[0].Earlystop_set(config['patience_adam']*U_PAT_FAC, config['delta_train_adam']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_adam']*V_PAT_FAC, config['delta_train_adam']/V_PAT_FAC)
        model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
        print("train adam start:")
        for iter in range(config['train_adam_epoch']):
            model[0].optimizer_adam.zero_grad()
            model[1].optimizer_adam.zero_grad()
            U = model[0](Ixy); V = model[1](Ixy)
            UV = torch.cat((U, V), dim=1)
            loss, mae = criterion_train_lgd(
                UV, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i]
                )
            loss.backward()
            model[0].optimizer_adam.step()
            model[1].optimizer_adam.step()
            config['epoch'] += 1
            model[0].Earlystop(mae, model[0], i , config['epoch'])
            model[1].Earlystop(mae, model[1], i , config['epoch'])
            if config['epoch']%config['print_feq'] == 1:
                epoch =  config['epoch'] 
                print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
            if model[0].early_stop or model[1].early_stop:
                if model[0].early_stop:
                    model[0].freeze_all_parameters()
                if model[1].early_stop:
                    model[1].freeze_all_parameters()
                if model[0].early_stop and model[1].early_stop:
                    print("train adam early stopping")
                    break
    if config['train_bfgs_epoch'] > config['max_iter']:
        model[0].Earlystop_set(config['patience_lbfgs']*U_PAT_FAC, config['delta_train_lbfgs']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_lbfgs']*V_PAT_FAC, config['delta_train_lbfgs']/V_PAT_FAC)
        model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
        print("train lbfgs start:") 
        for iter in range(config['train_bfgs_epoch']//config['max_iter']):
            def closure2_wrapper():
                loss, mae = closure2(
                    model, Ixy, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i], i
                    )
                if config['epoch']%config['print_feq'] == 1:
                    epoch =  config['epoch']
                    print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                return loss
            loss= closure2_wrapper()
            model[0].optimizer_lbfgs.step(closure2_wrapper)
            model[1].optimizer_lbfgs.step(closure2_wrapper)
            if model[0].early_stop or model[1].early_stop:
                if model[0].early_stop:
                    model[0].freeze_all_parameters()
                if model[1].early_stop:
                    model[1].freeze_all_parameters()
                if model[0].early_stop and model[1].early_stop:
                    print("train lbfgs early stopping")
                    break

def predict_stage(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv):
    model[0].eval()
    model[1].eval()
    with torch.no_grad():
        U = model[0](Ixy); V = model[1](Ixy)
    UV = torch.cat((U, V), dim=1)
    loss, mae = criterion_warmup_lgd(UV, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i])
    save_checkpoint(
        model, config['epoch'], 
        mae, config['model_path']+f"model{i+1:04d}.pth"
        )
    UV[:, 0] = UV[:, 0] * SCALE['scale'][i][0] + SCALE['scale'][i][2]
    UV[:, 1] = UV[:, 1] * SCALE['scale'][i][1] + SCALE['scale'][i][3]
    
    coords = XY_roi
    U = torch.zeros_like(RG).to(device)
    V = torch.zeros_like(RG).to(device)
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    U[y_coords, x_coords] = UV[:, 0]
    V[y_coords, x_coords] = UV[:, 1]
    uv[i,0,:,:] = U; uv[i,1,:,:] = V
    xyuv[i,:,0:2] = coords; xyuv[i,:,2:4] = UV
    return uv, xyuv

def frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv):
    print(f"Calculate the {i+1:04d}-th deformed image start:")
    model[0].train()
    model[1].train()
    print("warm up:")
    warm_up(i, Ixy, XY_roi, XY, RG, DG, ROI)
    print("train:")
    train_stage(i, Ixy, XY_roi, XY, RG, DG, ROI)
    uv, xyuv = predict_stage(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv)
    return uv, xyuv 

if __name__ == '__main__':
    
    img_dataset = lbdDataset(config['data_path'])
    RG, ROI, XY, XY_roi, Ixy = img_dataset.data_collect(device)
    train_loader = torch.utils.data.DataLoader(
        img_dataset, batch_size=config['Batchframes'], 
        shuffle=False, collate_fn=collate_fn)
    
    print(f"Simple FCN has {count_parameters(model[0])+count_parameters(model[1]):,} trainable parameters")
    
    batchframes = config['Batchframes']
    frames_list = list(range(batchframes))
    frames_list.remove(batchframes//2)
    H, L = RG.shape; N = len(img_dataset)
    total = XY_roi.shape[0]
    uv = torch.zeros((N, 2, H, L))
    xyuv = torch.zeros((N, total, 4))
    BATCH = math.floor(N / batchframes)
    
    print("train start")
    
    for batch, DGlist in enumerate(train_loader):
        if batch == BATCH:
            DG = DGlist[-1].to(device)
            model[0].unfreeze_and_initialize(init_type='xavier')
            model[1].unfreeze_and_initialize(init_type='xavier')
            # model[0].perturbation(perturbation_scale=0.001)
            # model[1].perturbation(perturbation_scale=0.001) 
            i = batch*batchframes + len(DGlist) - 1
            config['epoch'] = 0
            uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv)
            print("-------------*-------------")
            torch.cuda.empty_cache()
            for j, DG in enumerate(DGlist[:-1]):
                DG = DG.to(device)
                # model[0].unfreeze_and_initialize(init_type='xavier')
                # model[1].unfreeze_and_initialize(init_type='xavier')
                model[0].perturbation(perturbation_scale=0.001)
                model[1].perturbation(perturbation_scale=0.001) 
                i = batch*batchframes + j
                config['epoch'] = 0
                uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv)
                print("-------------*-------------")
                torch.cuda.empty_cache()
        else:
            DG = DGlist[batchframes//2].to(device)
            model[0].unfreeze_and_initialize(init_type='xavier')
            model[1].unfreeze_and_initialize(init_type='xavier')
            # model[0].perturbation(perturbation_scale=0.001)
            # model[1].perturbation(perturbation_scale=0.001)                     
            i = batch*batchframes + batchframes//2
            config['epoch'] = 0
            uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv)
            print("-------------*-------------")
            torch.cuda.empty_cache()
            for j in frames_list:
                DG = DGlist[j].to(device)
                # model[0].unfreeze_and_initialize(init_type='xavier')
                # model[1].unfreeze_and_initialize(init_type='xavier')
                model[0].perturbation(perturbation_scale=0.001)
                model[1].perturbation(perturbation_scale=0.001) 
                i = batch*batchframes + j
                config['epoch'] = 0
                uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv)
                print("-------------*-------------")
                torch.cuda.empty_cache()
    uv = uv.cpu().detach().numpy()
    xyuv = xyuv.cpu().detach().numpy()
    to_matlab(config['data_path'], 'result', uv)
    to_txt(config['data_path'], 'result', xyuv)
        
        
        
        

