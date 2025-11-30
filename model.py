import torch
import torch.nn as nn

# --- Constants for Network Input ---
# These are the dimensions of your preprocessed images
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

# --- AI Model Definition: DriverNet (NVIDIA CNN) ---

class DriverNet(nn.Module):

    def __init__(self):
        super(DriverNet, self).__init__()

        # Convolutional Layers: 5 layers to extract features from the image
        # Input size: 3x66x200 (YUV format)
        self.conv_layers = nn.Sequential(
            # 24@31x98
            nn.Conv2d(IMAGE_CHANNELS, 24, kernel_size=5, stride=2), 
            nn.ELU(),
            # 36@14x47
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            # 48@5x22
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            # 64@3x20
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            # 64@1x18
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        
        # Linear/Fully Connected Layers
        # The flattened input size is 64 * 1 * 18 = 1152
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 1 * 18, out_features=100),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=100, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1) # Output a single steering angle
        )
        

    def forward(self, input):
        # Permute the input image tensor from (B, H, W, C) to (B, C, H, W)
        input = input.permute(0, 3, 1, 2)
        
        output = self.conv_layers(input)
        
        # Flatten the output of the convolutional layers
        output = output.view(output.size(0), -1)
        
        # Pass through linear layers
        output = self.linear_layers(output)
        return output

# ---------------------------------------------------