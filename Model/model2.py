import torch
import torch.nn as nn

# --- Constants for Network Input ---
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

# --- Enhanced AI Model: DriverNet-Deep ---

class DriverNetDeep(nn.Module):

    def __init__(self):
        super(DriverNetDeep, self).__init__()

        # Convolutional Layers: 8 layers (increased from 5)
        self.conv_layers = nn.Sequential(
            # Block 1: Input 3x66x200 -> 24x33x100
            nn.Conv2d(IMAGE_CHANNELS, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            
            # Block 2: 24x33x100 -> 36x17x50
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            
            # Block 3: 36x17x50 -> 48x9x25
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            
            # Block 4: 48x9x25 -> 64x5x13
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            
            # Block 5: 64x5x13 -> 96x5x13 (same spatial size)
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ELU(),
            
            # Block 6: 96x5x13 -> 128x3x7
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            
            # Block 7: 128x3x7 -> 128x3x7 (same spatial size)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            
            # Block 8: 128x3x7 -> 128x2x4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        
        # CRITICAL: Flattened size = 128 channels * 2 height * 4 width = 1024
        # Linear/Fully Connected Layers: 6 layers (increased from 4)
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),  # <-- MUST BE 1024
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            
            nn.Linear(in_features=64, out_features=32),
            nn.ELU(),
            
            nn.Linear(in_features=32, out_features=1)  # Output steering angle
        )
        

    def forward(self, input):
        # Permute from (batch, height, width, channels) to (batch, channels, height, width)
        input = input.permute(0, 3, 1, 2)
        
        # Pass through convolutional layers
        output = self.conv_layers(input)
        
        # Flatten for linear layers
        output = output.reshape(output.size(0), -1)
        
        # Pass through linear layers
        output = self.linear_layers(output)
        
        return output


# Test function to verify dimensions
def test_model():
    """Test the model with a dummy input to verify dimensions."""
    model = DriverNetDeep()
    
    # Create a dummy input: batch_size=4, height=66, width=200, channels=3
    dummy_input = torch.randn(4, 66, 200, 3)
    
    print("Testing DriverNetDeep...")
    print(f"Input shape: {dummy_input.shape}")
    print(f"First linear layer expects: {model.linear_layers[0].in_features} features")
    
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("✅ Model architecture is correct!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()