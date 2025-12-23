import torch
import torch.nn as nn
import torch.nn.init as init

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        
        # Block 1: Conv(3x3, pad 1, 175) -> MLP(1x1, 2240) -> MaxPool(2x2)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 175, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(175, 2240, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2) # 40x40 -> 20x20
        )
        
        # Block 2: Conv(3x3, pad 1, 175) -> MLP(1x1, 640) -> MaxPool(2x2)
        self.block2 = nn.Sequential(
            nn.Conv2d(2240, 175, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(175, 640, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2) # 20x20 -> 10x10
        )
        
        # Block 3: Conv(3x3, pad 1, 175) -> MLP(1x1, 320) -> MaxPool(2x2)
        self.block3 = nn.Sequential(
            nn.Conv2d(640, 175, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(175, 320, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2) # 10x10 -> 5x5
        )
        
        # Output layer
        # Final feature map size is 320 channels * 5 * 5 = 8000
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(320 * 5 * 5, num_classes)
        
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
        x = self.flatten(x)
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
