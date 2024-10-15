import os
import torch
import numpy as np
import random
import sys
import math
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

from debug_until_fun import debug_plot

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
        model[0].Earlystop_set(config['patience_adam']*U_PAT_FAC, config['delta_warm_adam']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_adam']*V_PAT_FAC, config['delta_warm_adam']/V_PAT_FAC)
        if Keyframe_FALG:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['warm_adam_epoch']
        else:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['warm_adam_epoch']//2
            # model[0].freeze_layers(); model[1].freeze_layers()
        print("warm adam start:")
        file.write("warm adam start:\n")
        for iter in range(epoch_optim):
            model[0].optimizer_adam.zero_grad()
            model[1].optimizer_adam.zero_grad()
            U = model[0](Ixy); V = model[1](Ixy)
            UV = torch.cat((U, V), dim=1)
            # if i != 2:
            #     debug_plot(model, Ixy, XY_roi, ROI, SCALE, i)
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
                file.write(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}\n")
            if model[0].early_stop or model[1].early_stop:
                if model[0].early_stop:
                    model[0].freeze_all_parameters()
                if model[1].early_stop:
                    model[1].freeze_all_parameters()
                if model[0].early_stop and model[1].early_stop:
                    print("warm adam early stopping")
                    file.write("warm adam early stopping\n")
                    break
    if config['warm_bfgs_epoch'] > config['max_iter']:
        model[0].Earlystop_set(config['patience_lbfgs']*U_PAT_FAC, config['delta_warm_lbfgs']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_lbfgs']*V_PAT_FAC, config['delta_warm_lbfgs']/V_PAT_FAC)
        if Keyframe_FALG:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['warm_bfgs_epoch']
        else:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['warm_bfgs_epoch']//2
            # model[0].freeze_layers(); model[1].freeze_layers()
        print("warm lbfgs start:") 
        file.write("warm lbfgs start:\n")
        for iter in range(epoch_optim//config['max_iter']):
            def closure1_wrapper():
                loss, mae = closure1(
                    model, Ixy, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i], i
                    )
                if config['epoch']%config['print_feq'] == 1:
                    epoch =  config['epoch'] 
                    print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                    file.write(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}\n")
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
                    print("warm lbfgs early stopping")
                    file.write("warm lbfgs early stopping\n")
                    break
            
def train_stage(i, Ixy, XY_roi, XY, RG, DG, ROI):
    if config['train_adam_epoch'] != 0:
        # model[0].optimizer_adam.param_groups[0]['lr'] = config['train_lr']
        # model[1].optimizer_adam.param_groups[0]['lr'] = config['train_lr']
        model[0].reset_optim(config); model[1].reset_optim(config)
        model[0].Earlystop_set(config['patience_adam']*U_PAT_FAC, config['delta_train_adam']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_adam']*V_PAT_FAC, config['delta_train_adam']/V_PAT_FAC)
        if Keyframe_FALG:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['train_adam_epoch']
        else:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['train_adam_epoch']//2
            # model[0].freeze_layers(); model[1].freeze_layers()
        print("train adam start:")
        file.write("train adam start:\n")
        for iter in range(epoch_optim):
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
                file.write(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}\n")
            if model[0].early_stop or model[1].early_stop:
                if model[0].early_stop:
                    model[0].freeze_all_parameters()
                if model[1].early_stop:
                    model[1].freeze_all_parameters()
                if model[0].early_stop and model[1].early_stop:
                    print("train adam early stopping")
                    file.write("train adam early stopping\n")
                    break
    if config['train_bfgs_epoch'] > config['max_iter']:
        model[0].Earlystop_set(config['patience_lbfgs']*U_PAT_FAC, config['delta_train_lbfgs']/U_PAT_FAC)
        model[1].Earlystop_set(config['patience_lbfgs']*V_PAT_FAC, config['delta_train_lbfgs']/V_PAT_FAC)
        if Keyframe_FALG:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['train_bfgs_epoch']
        else:
            model[0].unfreeze_all_parameters(); model[1].unfreeze_all_parameters()
            epoch_optim = config['train_bfgs_epoch']//2
            # model[0].freeze_layers(); model[1].freeze_layers()
        print("train lbfgs start:") 
        file.write("train lbfgs start:\n")
        for iter in range(epoch_optim//config['max_iter']):
            def closure2_wrapper():
                loss, mae = closure2(
                    model, Ixy, XY_roi, XY, RG, DG, ROI, SCALE['scale'][i], i
                    )
                if config['epoch']%config['print_feq'] == 1:
                    epoch =  config['epoch']
                    print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                    file.write(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}\n")
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
                    file.write("train lbfgs early stopping\n")
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
    file.write(f"Calculate the {i+1:04d}-th deformed image start:\n")
    model[0].train(); model[1].train()
    model[0].get_optimizer(config); model[1].get_optimizer(config)
    print("warm up:")
    file.write("warm up:\n")
    warm_up(i, Ixy, XY_roi, XY, RG, DG, ROI)
    print("train:")
    file.write("train:\n")
    train_stage(i, Ixy, XY_roi, XY, RG, DG, ROI)
    uv, xyuv = predict_stage(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, j)
    return uv, xyuv 

if __name__ == '__main__':
        
    img_dataset = lbdDataset(config['data_path'])
    RG, ROI, XY, XY_roi, Ixy = img_dataset.data_collect(device)
    train_loader = torch.utils.data.DataLoader(
        img_dataset, batch_size=config['Batchframes'], 
        shuffle=False, collate_fn=collate_fn)
    
    print(f"Simple FCNN has {count_parameters(model[0])+count_parameters(model[1]):,} trainable parameters")
    
    batchframes = config['Batchframes']
    frames_list = list(range(batchframes))
    frames_list.remove(batchframes//2)
    H, L = RG.shape; N = len(img_dataset)
    total = XY_roi.shape[0]
    BATCH = math.floor(N / batchframes)
    Keyframe_FALG = True
    
    print("train start")
    file = open(config['log_path']+'training_log.txt', 'w')
    for batch, DGlist in enumerate(train_loader):
        if batch == BATCH:
            uv = torch.zeros((len(DGlist), 2, H, L))
            xyuv = torch.zeros((len(DGlist), total, 4))
            DG = DGlist[-1].to(device)
            Keyframe_FALG = True
            model[0].unfreeze_and_initialize(init_type='xavier')
            model[1].unfreeze_and_initialize(init_type='xavier')
            i = batch*batchframes + len(DGlist) - 1
            config['epoch'] = 0
            uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, len(DGlist)-1)
            print("-------------*-------------")
            file.write("-------------*-------------\n")
            torch.cuda.empty_cache()
            Keyframe_FALG = False
            for j, DG in enumerate(DGlist[:-1]):
                DG = DG.to(device)
                # model[0].perturbation(perturbation_scale=0.001)
                # model[1].perturbation(perturbation_scale=0.001)
                i = batch*batchframes + j
                config['epoch'] = 0
                uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, j)
                print("-------------*-------------")
                file.write("-------------*-------------\n")
                torch.cuda.empty_cache()
            uv = uv.cpu().detach().numpy()
            xyuv = xyuv.cpu().detach().numpy()
            to_matlab(config['data_path'], f'mat{BATCH*batchframes+1:03d}-{BATCH*batchframes+len(DGlist):03d}', uv)
            to_txt(config['data_path'], f'txt{BATCH*batchframes+1:03d}-{BATCH*batchframes+len(DGlist):03d}', xyuv)
            torch.cuda.empty_cache()
        else:
            uv = torch.zeros((len(DGlist), 2, H, L))
            xyuv = torch.zeros((len(DGlist), total, 4))
            DG = DGlist[batchframes//2].to(device)
            Keyframe_FALG = True
            model[0].unfreeze_and_initialize(init_type='xavier')
            model[1].unfreeze_and_initialize(init_type='xavier')                   
            i = batch*batchframes + batchframes//2
            config['epoch'] = 0
            uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, batchframes//2)
            print("-------------*-------------")
            file.write("-------------*-------------\n")
            torch.cuda.empty_cache()
            Keyframe_FALG = False
            for j in frames_list:
                DG = DGlist[j].to(device)
                # model[0].perturbation(perturbation_scale=0.001)
                # model[1].perturbation(perturbation_scale=0.001)
                i = batch*batchframes + j
                config['epoch'] = 0
                uv, xyuv = frame_calculate(i, Ixy, XY_roi, XY, RG, DG, ROI, uv, xyuv, j)
                print("-------------*-------------")
                file.write("-------------*-------------\n")
                torch.cuda.empty_cache()
            uv = uv.cpu().detach().numpy()
            xyuv = xyuv.cpu().detach().numpy()
            to_matlab(config['data_path'], f'mat{batch*batchframes+1:03d}-{batch*batchframes+len(DGlist):03d}', uv)
            to_txt(config['data_path'], f'txt{batch*batchframes+1:03d}-{batch*batchframes+len(DGlist):03d}', xyuv)
            torch.cuda.empty_cache()
    file.close()
        
        
        
        