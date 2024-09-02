import os
import torch
import numpy as np
import random
import sys
sys.path.append("./layers")
sys.path.append("./utils")

# from layers.criterion import criterion_warmup, criterion_train
# from layers.EarlyStop import EarlyStopping
# from layers.FCNN import DNN
# from utils.utils import save_checkpoint
# from utils.readData import lbdDataset, collate_fn
# from utils.result_plot import to_matlab

from criterion import criterion_warmup, criterion_train
from EarlyStop import EarlyStopping
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

seed_everything(1234)
model = DNN(config['layers']).to(device)
max_iter=config['max_iter']; max_eval = int(1.25 * max_iter)

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1, max_iter=max_iter, max_eval=max_eval,
    history_size=50, tolerance_grad=1e-06,
    tolerance_change=5e-06,
    line_search_fn="strong_wolfe")

optimizer_adam = torch.optim.Adam(
    model.parameters(), lr=config['warm_lr'],  eps=1e-8)

early_stop_adam = EarlyStopping(
        patience=config['patience_adam'], delta=config['delta_warm_adam'], 
        path='./weights/checkpoint/checkpoint_adam.pth')
early_stop_lbfgs = EarlyStopping(
        patience=config['patience_lbfgs'], delta=config['delta_warm_lbfgs'], 
        path='./weights/checkpoint/checkpoint_lbfgs.pth')

def closure1(model, XY_roi, XY, Iref, Idef, ROI, scale):
    optimizer_lbfgs.zero_grad()
    UV = model(Ixy)
    loss, mae = criterion_warmup(UV, XY_roi, XY, Iref, Idef, ROI, scale)
    loss.backward()
    config['epoch'] += 1
    early_stop_lbfgs(mae, model, optimizer_lbfgs)
    return loss, mae

def closure2(model, XY_roi, XY, Iref, Idef, ROI, scale):
    optimizer_lbfgs.zero_grad()
    UV = model(Ixy)
    loss, mae = criterion_train(UV, XY_roi, XY, Iref, Idef, ROI, scale)
    loss.backward()
    config['epoch'] += 1
    early_stop_lbfgs(mae, model, optimizer_lbfgs)
    return loss, mae
    
     
if __name__ == '__main__':
    
    img_dataset = lbdDataset(config['data_path'])
    RG, ROI, XY, XY_roi, Ixy = img_dataset.data_collect(device)
    train_loader = torch.utils.data.DataLoader(
        img_dataset, batch_size=1, 
        shuffle=False, collate_fn=collate_fn)
    
    print(f"Simple FCN has {count_parameters(model):,} trainable parameters")
    
    H, L = RG.shape; N = len(train_loader)
    total = XY_roi.shape[0]
    uv = torch.zeros((N, 2, H, L))
    xyuv = torch.zeros((N, total, 4))
    
    print("train start")
    for i, DG in enumerate(train_loader):
        DG = DG[0].to(device)
        model.train()
        
        print(f"Calculate the {i+1:04d}-th deformed image start:")
        print("warm up:")
        early_stop_adam.path  = f"./weights/checkpoint/example{i+1:04d}_warm_adam.pth"
        early_stop_lbfgs.path = f"./weights/checkpoint/example{i+1:04d}_warm_lbfgs.pth"
        early_stop_adam.delta = config['delta_warm_adam']
        early_stop_lbfgs.delta = config['delta_warm_lbfgs']
        optimizer_adam.param_groups[0]['lr'] = config['warm_lr']
        if config['warm_adam_epoch'] != 0:
            print("warm adam start:")
            for iter in range(config['warm_adam_epoch']):
                optimizer_adam.zero_grad()
                UV = model(Ixy)
                loss, mae = criterion_warmup(UV, XY_roi, XY, RG, DG, ROI, config['scale'][i])
                loss.backward()
                optimizer_adam.step()
                config['epoch'] += 1
                early_stop_adam(mae, model, optimizer_adam)
                if config['epoch']%config['print_feq'] == 1:
                    epoch =  config['epoch'] 
                    print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                if early_stop_adam.early_stop:
                    print("warm adam early stopping")
                    early_stop_adam.Reset()
                    break
        if config['warm_bfgs_epoch'] > config['max_iter']:
            print("warm lbfgs start:") 
            for iter in range(config['warm_bfgs_epoch']//config['max_iter']):
                def closure1_wrapper():
                    loss, mae = closure1(model, XY_roi, XY, RG, DG, ROI, config['scale'][i])
                    if config['epoch']%config['print_feq'] == 1:
                        epoch =  config['epoch'] 
                        print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                    return loss
                loss = closure1_wrapper()
                optimizer_lbfgs.step(closure1_wrapper)
                if early_stop_lbfgs.early_stop:
                    print("warm lbfgs early stopping")
                    early_stop_lbfgs.Reset()
                    break
        
        print("train:")
        early_stop_adam.path  = f"./weights/checkpoint/example{i:04d}_train_adam.pth"
        early_stop_lbfgs.path = f"./weights/checkpoint/example{i:04d}_train_lbfgs.pth"
        early_stop_adam.delta = config['delta_train_adam']
        early_stop_lbfgs.delta = config['delta_train_lbfgs']
        if config['train_adam_epoch'] != 0:
            print("train adam start:")
            optimizer_adam.param_groups[0]['lr'] = config['train_lr']
            for iter in range(config['train_adam_epoch']):
                optimizer_adam.zero_grad()
                UV = model(Ixy)
                loss, mae = criterion_warmup(UV, XY_roi, XY, RG, DG, ROI, config['scale'][i])
                loss.backward()
                optimizer_adam.step()
                config['epoch'] += 1
                early_stop_adam(mae, model, optimizer_adam)
                if config['epoch']%config['print_feq'] == 1:
                    epoch =  config['epoch'] 
                    print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                if early_stop_adam.early_stop:
                    print("train adam early stopping")
                    early_stop_adam.Reset()
                    break
        if config['train_bfgs_epoch'] > config['max_iter']:
            print("warm lbfgs start:") 
            for iter in range(config['warm_bfgs_epoch']//config['max_iter']):
                def closure2_wrapper():
                    loss, mae = closure2(model, XY_roi, XY, RG, DG, ROI, config['scale'][i])
                    if config['epoch']%config['print_feq'] == 1:
                        epoch =  config['epoch']
                        print(f"Epoch [{epoch:4d}], MAE: {mae.item():.5f}")
                    return loss
                loss= closure2_wrapper()
                optimizer_lbfgs.step(closure2_wrapper)
                if early_stop_lbfgs.early_stop:
                    print("train lbfgs early stopping")
                    early_stop_lbfgs.Reset()
                    break
        print("-------------*-------------")
        
        model.eval()
        UV = model(Ixy)
        loss, mae = criterion_warmup(UV, XY_roi, XY, RG, DG, ROI, config['scale'][i])
        save_checkpoint(
            model, optimizer_adam, optimizer_lbfgs, config['epoch'], 
            mae, config['model_path']+f"model{i+1:04d}.pth"
            )
        UV[:, 0] = UV[:, 0] * config['scale'][i][0]
        UV[:, 1] = UV[:, 1] * config['scale'][i][1]
        
        coords = XY_roi
        U = torch.zeros_like(RG).to(device)
        V = torch.zeros_like(RG).to(device)
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        U[y_coords, x_coords] = UV[:, 0]
        V[y_coords, x_coords] = UV[:, 1]
        uv[i,0,:,:] = U; uv[i,1,:,:] = V
        xyuv[i,:,0:2] = coords; xyuv[i,:,2:4] = UV
    
    uv = uv.cpu().detach().numpy()
    xyuv = xyuv.cpu().detach().numpy()
    to_matlab(config['data_path'], 'result', uv)
    to_txt(config['data_path'], 'result', xyuv)
        
        
        
        