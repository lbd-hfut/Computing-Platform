import torch
from criterion import criterion_warmup, criterion_train
from EarlyStop import EarlyStopping

def warmup(model, optimizer1, optimizer2, RG, DG, ROI, XY, XY_roi, Ixy, epoch, i, config):
    model.train()
    
    for iter in range(config['warm_adam_epoch']):
        optimizer1.zero_grad()
        UV = model(Ixy)
        loss, mae = criterion_warmup(UV, XY_roi, XY, RG, DG, ROI, config['scale'][i])
        loss.backward()
        optimizer1.step()
        
    for iter in range(config['warm_bfgs_epoch']//config['max_iter']):
        optimizer2.zero_grad()
        UV = model(Ixy)
        loss = criterion_warmup(UV, XY_roi, XY, RG, DG, ROI, config['scale'][i])
        loss.backward()
        optimizer2.step()