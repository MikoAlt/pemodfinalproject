import torch
import torch.nn as nn
import torch.nn.init as init

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        

        # Target ~5.2M parameters        
        # Block 1: 40x40 -> 20x20
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 20x20 -> 10x10
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 10x10 -> 5x5
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully Connected Layers
        # 5x5 * 256 = 6400
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 758),
            nn.LeakyReLU(0.01),
            nn.Linear(758, num_classes)
        )
        
        # He Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (B, 1, 40, 40)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
        print(f"CNN Model saved to {filepath}")

    @staticmethod
    def load_model(filepath, input_channels, num_classes):
        model = CNNModel(input_channels, num_classes)
        model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        model.eval()
        print(f"CNN Model loaded from {filepath}")
        return model
