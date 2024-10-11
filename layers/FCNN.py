import torch
import torch.nn as nn

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
        
        # # Define hidden layers (MLP)
        # self.hidden_layers = nn.ModuleList(
        #     [nn.Linear(self.width[i], self.width[i+1]) for i in range(self.num_layers-1)]
        #     )
        
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
        x = self.input_ln(x)
        # x = 5 * self.a[0] * x
        x = torch.tanh(x)
        # Hidden layers (MLP)
        for i in range(self.num_layers-1):
            x = self.hidden_layers[i](x)
            # x = self.hidden_lns[i](x)
            # x = 5 * self.a[i + 1] * x
            x = torch.tanh(x)
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
    
    # Set He Kaiming initialization, only applied to the specified layers
    def set_kaiming_initialization(self):
        # nn.init.kaiming_normal_(self.hidden_layers[-1], nonlinearity='relu')
        # if self.hidden_layers[-1].bias is not None:
        #     nn.init.constant_(self.hidden_layers[-1].bias, 0)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

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
                
            # # Initialize LayerNorm parameters
            # for ln in self.hidden_lns:
            #     nn.init.kaiming_normal_(ln.weight)
            #     if ln.bias is not None:
            #         nn.init.constant_(ln.bias, 0)
    
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
                
            # # Initialize LayerNorm parameters
            # for ln in self.hidden_lns:
            #     nn.init.xavier_normal_(ln.weight)
            #     if ln.bias is not None:
            #         nn.init.constant_(ln.bias, 0)
        
        # Initialize LayerNorm parameters
        for ln in self.hidden_lns:
            nn.init.ones_(ln.weight)  # 初始化为1
            nn.init.zeros_(ln.bias)   # 初始化为0
    



