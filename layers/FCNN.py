import torch
import torch.nn as nn
import sys

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.width = layers[1:-1]
        self.num_layers = len(self.width)
        self.input_size = layers[0]
        self.output_size = layers[-1]
        
        # Define input layer
        self.input_layer = nn.Linear(self.input_size, self.width[0])
        self.input_ln = nn.LayerNorm(self.width[0])
        
        # Define hidden layers (MLP)  
        self.hidden_layers = nn.ModuleList()  
        self.hidden_lns = nn.ModuleList()  
        for i in range(self.num_layers-1):  
            self.hidden_layers.append(nn.Linear(self.width[i], self.width[i+1]))  
            self.hidden_lns.append(nn.LayerNorm(self.width[i+1]))
        
        # Define output layer
        self.output_layer = nn.Linear(self.width[-1], self.output_size)
        
        # Define activation function parameter 'a'
        self.a = nn.Parameter(torch.tensor([0.2] * (self.num_layers + 1)))

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        # x = self.input_ln(x)
        # x = 5 * self.a[0] * x
        x = torch.tanh(x)
        # Hidden layers (MLP)
        for i in range(self.num_layers-1):
            x = self.hidden_layers[i](x)
            # x = self.hidden_lns[i](x)
            # x = 5 * self.a[i + 1] * x
            x = torch.relu(x)
        # Output layer
        # x = 5 * self.a[-1] * x
        x = self.output_layer(x)
        return x
    
    # Freeze Specified Layers
    def freeze_layers(self):
        # for param in self.input_layer.parameters():
        #     param.requires_grad = False
        for i in range(self.num_layers-1):
            for param in self.hidden_layers[i].parameters():
                param.requires_grad = False
        # self.a[:-2].detach()
        self.a[:-1].detach()
        
    def perturbation(self, perturbation_scale=0.05):
        with torch.no_grad():
            ksi_weight = torch.zeros_like(self.output_layer.weight)
            nn.init.normal_(ksi_weight, mean=0.0, std=perturbation_scale)
            self.output_layer.weight.add_(ksi_weight)
            if self.output_layer.bias is not None:
                ksi_bias = torch.zeros_like(self.output_layer.bias)
                nn.init.normal_(ksi_bias, mean=0.0, std=perturbation_scale)
                self.output_layer.bias.add_(ksi_bias)

    # Freeze all trainable parameters  
    def freeze_all_parameters(self):  
        for param in self.parameters():  
            param.requires_grad = False  
  
    # Unfreeze all trainable parameters  
    def unfreeze_all_parameters(self):  
        for param in self.parameters():  
            param.requires_grad = True
            
    # Unfreeze all layers and apply He Kaiming initialization
    def unfreeze_and_initialize(self, init_type='xavier'):
        # Unfreeze all layers
        for param in self.parameters():
            if not param.requires_grad:
                param.requires_grad = True
        
        # Apply selected initialization
        if init_type == 'kaiming':
            # Apply He Kaiming initialization to input layer, hidden layers, and output layer
            nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
            if self.input_layer.bias is not None:
                nn.init.constant_(self.input_layer.bias, 0)
            
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            
            nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
            if self.output_layer.bias is not None:
                nn.init.constant_(self.output_layer.bias, 0)
    
        elif init_type == 'xavier':
            # Apply Xavier initialization to input layer, hidden layers, and output layer
            nn.init.xavier_normal_(self.input_layer.weight)
            if self.input_layer.bias is not None:
                nn.init.constant_(self.input_layer.bias, 0)
            
            for layer in self.hidden_layers:
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            
            nn.init.xavier_normal_(self.output_layer.weight)
            if self.output_layer.bias is not None:
                nn.init.constant_(self.output_layer.bias, 0)
        
        # Initialize LayerNorm parameters
        for ln in self.hidden_lns:
            nn.init.ones_(ln.weight)  # 初始化为1
            nn.init.zeros_(ln.bias)   # 初始化为0
        

    
    # Set the parameters for the early stop method
    def Earlystop_set(self, patience=10, delta=0, path=None):
        self.patience = int(patience)
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    # Logical judgment of executing the early stop method
    def Earlystop(self, val_loss, model, i, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, i, epoch)
        elif self.best_loss < val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, i, epoch)
            self.counter = 0
            
    # Save the checkpoint of the current optimal model
    def save_checkpoint(self, val_loss, model, i, epoch):
        if self.path:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'loss': val_loss
            }
            torch.save(checkpoint, self.path+f'checkpoint_exp{i}_epoch{epoch}.pth')
            self.best_model = model.state_dict()
        
    # Reset the relevant parameters of the early stop method
    def Earlystop_Reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    # Obtain optimizer based on configuration
    def get_optimizer(self, config):
        # Check if the configuration dictionary contains the necessary parameters
        if 'max_iter' not in config:  
            raise ValueError(
                "Config dictionary must contain 'max_iter' for LBFGS optimizer."
                )
        elif 'warm_lr' not in config:
            raise ValueError(
                "Config dictionary must contain 'warm_lr' for Adam optimizer."
                )
        #Initialize LBFGS optimizer
        max_iter=config['max_iter']; max_eval = int(1.25 * max_iter)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.parameters(), lr=1, max_iter=max_iter, max_eval=max_eval,
            history_size=50, tolerance_grad=1e-06,
            tolerance_change=5e-06,
            line_search_fn="strong_wolfe")
        # Initialize Adam optimizer
        self.optimizer_adam = torch.optim.Adam(
            self.parameters(), lr=config['warm_lr'],  eps=1e-8, weight_decay=config['weight_decay'])
        
    def reset_optim(self, config):
        #Initialize LBFGS optimizer
        max_iter=config['max_iter']; max_eval = int(1.25 * max_iter)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.parameters(), lr=1, max_iter=max_iter, max_eval=max_eval,
            history_size=50, tolerance_grad=1e-06,
            tolerance_change=5e-06,
            line_search_fn="strong_wolfe")
        # Initialize Adam optimizer
        self.optimizer_adam = torch.optim.Adam(
            self.parameters(), lr=config['warm_lr'],  eps=1e-8, weight_decay=config['weight_decay'])
    
    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=20, eta_min=1e-6)
        


