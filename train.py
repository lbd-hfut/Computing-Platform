import os
import torch
import numpy as np
import random
import sys
from scipy.io import loadmat
sys.path.append("./layers")
sys.path.append("./utils")

from criterion import criterion_warmup, criterion_train
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
    loss, mae = criterion_warmup(UV, XY_roi, XY, Iref, Idef, ROI, scale)
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
    loss, mae = criterion_train(UV, XY_roi, XY, Iref, Idef, ROI, scale)
    loss.backward()
    config['epoch'] += 1
    model[0].Earlystop(mae, model[0], i , config['epoch'])
    model[1].Earlystop(mae, model[1], i , config['epoch'])
    return loss, mae

def warm_up(i, Ixy, XY_roi, XY, RG, DG, ROI):
    if config['warm_adam_epoch'] != 0:
        model[0].Earlystop_set(config['patience_adam'], config['delta_warm_adam'])
        model[1].Earlystop_set(config['patience_adam'], config['delta_warm_adam'])
        print("warm adam start:")
        for iter in range(config['warm_adam_epoch']):
            model[0].optimizer_adam.zero_grad()
            model[1].optimizer_adam.zero_grad()
            U = model[0](Ixy); V = model[1](Ixy)
            UV = torch.cat((U, V), dim=1)
            loss, mae = criterion_warmup(
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
            if model[0].early_stop and model[1].early_stop:
                print("warm adam early stopping")
                break
    if config['warm_bfgs_epoch'] > config['max_iter']:
        model[0].Earlystop_set(config['patience_lbfgs'], config['delta_warm_lbfgs'])
        model[1].Earlystop_set(config['patience_lbfgs'], config['delta_warm_lbfgs'])
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
            if model[0].early_stop and model[1].early_stop:
                print("warm lbfgs early stopping")
                break
            
def train_stage(i, Ixy, XY_roi, XY, RG, DG, ROI):
    if config['train_adam_epoch'] != 0:
        model[0].optimizer_adam.param_groups[0]['lr'] = config['train_lr']
        model[1].optimizer_adam.param_groups[0]['lr'] = config['train_lr']
        model[0].Earlystop_set(config['patience_adam'], config['delta_train_adam'])
        model[1].Earlystop_set(config['patience_adam'], config['delta_train_adam'])
        print("train adam start:")
        for iter in range(config['train_adam_epoch']):
            model[0].optimizer_adam.zero_grad()
            model[1].optimizer_adam.zero_grad()
            U = model[0](Ixy); V = model[1](Ixy)
            UV = torch.cat((U, V), dim=1)
            loss, mae = criterion_train(
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
            if model[0].early_stop and model[1].early_stop:
                print("train adam early stopping")
                break
    if config['train_bfgs_epoch'] > config['max_iter']:
        model[0].Earlystop_set(config['patience_lbfgs'], config['delta_train_lbfgs'])
        model[1].Earlystop_set(config['patience_lbfgs'], config['delta_train_lbfgs'])
        print("warm lbfgs start:") 
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
            if model[0].early_stop and model[1].early_stop:
                print("train lbfgs early stopping")
                break

def predict_stage(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, j):
    model[0].eval()
    model[1].eval()
    with torch.no_grad():
        U = model[0](Ixy); V = model[1](Ixy)
    UV = torch.cat((U, V), dim=1)
    loss, mae = criterion_warmup(UV, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i])
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
    uv[j,0,:,:] = U; uv[j,1,:,:] = V
    xyuv[j,:,0:2] = coords; xyuv[j,:,2:4] = UV
    return uv, xyuv

def frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, j):
    print(f"Calculate the {i+1:04d}-th deformed image start:")
    model[0].train()
    model[1].train()
    print("warm up:")
    warm_up(i, Ixy, XY_roi, XY, RG, DG, ROI)
    print("train:")
    train_stage(i, Ixy, XY_roi, XY, RG, DG, ROI)
    uv, xyuv = predict_stage(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, j)
    return uv, xyuv 
     
if __name__ == '__main__':
    img_dataset = lbdDataset(config['data_path'])
    RG, ROI, XY, XY_roi, Ixy = img_dataset.data_collect(device)
    train_loader = torch.utils.data.DataLoader(
        img_dataset, batch_size=1, 
        shuffle=False, collate_fn=collate_fn)
    
    print(f"Simple FCN has {count_parameters(model[0])+count_parameters(model[1]):,} trainable parameters")
    
    H, L = RG.shape; N = len(train_loader)
    total = XY_roi.shape[0]
    uv = torch.zeros((N, 2, H, L))
    xyuv = torch.zeros((N, total, 4))
    
    print("train start")
    for i, DG_list in enumerate(train_loader):
        uv = torch.zeros((1, 2, H, L))
        xyuv = torch.zeros((1, total, 4))
        DG = DG_list[0].to(device)
        model[0].unfreeze_and_initialize()
        model[1].unfreeze_and_initialize()
        config['epoch'] = 0
        uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, 0)
        torch.cuda.empty_cache()
        print("-------------*-------------")
        uv = uv.cpu().detach().numpy()
        xyuv = xyuv.cpu().detach().numpy()
        to_matlab(config['data_path'], f'mat{i+1:03d}', uv)
        to_txt(config['data_path'], f'txt{i+1:03d}', xyuv)
        torch.cuda.empty_cache()
        
        
        
        