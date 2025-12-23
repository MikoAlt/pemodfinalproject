import torch
import torch.nn as nn
import torch.nn.init as init

class MatrixModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        hidden_sizes: list of integers [h1, h2, h3]
        """
        super(MatrixModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.layers.append(nn.Linear(hidden_sizes[1], hidden_sizes[2]))
        self.layers.append(nn.Linear(hidden_sizes[2], output_size))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # He Initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        # x shape: (batch_size, 1600)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.leaky_relu(x)
        
        # Output layer
        x = self.layers[-1](x)
        # We don't use Softmax here as CrossEntropyLoss includes it
        return x

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        print(f"Model loaded from {filepath}")
