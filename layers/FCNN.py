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
        
        # Define hidden layers (MLP)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.width[i], self.width[i+1]) for i in range(self.num_layers-1)]
            )
        
        # Define output layer
        self.output_layer = nn.Linear(self.width[-1], self.output_size)
        
        
        # Define activation function parameter 'a'
        self.a = nn.Parameter(torch.tensor([0.2] * (self.num_layers + 2)))

    def forward(self, x):
        
        # Input layer
        x = self.input_layer(x)
        x = 5 * self.a[0] * x
        x = torch.tanh(x)
        
        # Hidden layers (MLP)
        for i in range(self.num_layers-1):
            x = self.hidden_layers[i](x)
            x = 5 * self.a[i + 1] * x
            x = torch.tanh(x)
        
        # Output layer
        x = 5 * self.a[-1] * x
        x = self.output_layer(x)
        
        return x
    
    # Freeze Specified Layers
    def freeze_layers(self, layer_names):
        for name, param in self.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False
                print(f"Layer {name} has been frozen.")

    # Modify Specified Layers
    def modify_output_layer(self, new_output_size):
        self.output_size = new_output_size
        self.output_layer = nn.Linear(self.width[-1], self.output_size)
        print(f"Output layer modified to output {new_output_size} units.")
    
    # Set He Kaiming initialization, only applied to the specified layers
    def set_kaiming_initialization(self):
        nn.init.kaiming_normal_(self.hidden_layers[-1], nonlinearity='relu')
        if self.hidden_layers[-1].bias is not None:
            nn.init.constant_(self.hidden_layers[-1].bias, 0)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)


    



