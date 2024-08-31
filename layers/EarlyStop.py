import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='./weights/checkpoint/checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        self.best_optimizer = None

    def __call__(self, val_loss, model, optimizer):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
        elif self.best_loss < val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }
        torch.save(checkpoint, self.path)
        self.best_model = model.state_dict()
        self.best_optimizer = optimizer.state_dict()
        
    def Reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
