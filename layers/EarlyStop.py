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
        elif val_loss > self.best_loss + self.delta:
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
        


# # 使用 EarlyStopping
# early_stopping = EarlyStopping(patience=10, delta=0.01)

# for epoch in range(epochs):
#     # 训练代码...
#     val_loss = compute_validation_loss()
#     # 检查是否需要早停
#     early_stopping(val_loss, model)
#     if early_stopping.early_stop:
#         print("Early stopping")
#         break

# # 加载最佳模型
# model.load_state_dict(early_stopping.best_model)
