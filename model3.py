import torch
import torch.nn as nn
import torch.optim as optim

# --- Constants for Network Input ---
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

# --- Residual Block with Skip Connections ---
class ResidualBlock(nn.Module):
    """Residual block with skip connections for better gradient flow"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.elu(out)
        
        return out


# --- Spatial Attention Module ---
class SpatialAttention(nn.Module):
    """Spatial attention to focus on important road features"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


# --- Ultra Deep Model ---
class DriverNetUltra(nn.Module):

    def __init__(self):
        super(DriverNetUltra, self).__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        
        # Stage 1: Residual blocks with attention
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            SpatialAttention()
        )
        
        # Stage 2: Deeper features
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 96, stride=1),
            SpatialAttention()
        )
        
        # Stage 3: High-level features - Added one more residual block for complexity
        self.stage3 = nn.Sequential(
            ResidualBlock(96, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),  # Additional block
            SpatialAttention()
        )
        
        # Stage 4: Abstract features - Added one more residual block
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 160, stride=2),
            ResidualBlock(160, 160, stride=1),
            ResidualBlock(160, 160, stride=1),
            ResidualBlock(160, 160, stride=1)  # Additional block
        )
        
        # Stage 5: Final conv layers - Added one more conv layer
        self.stage5 = nn.Sequential(
            nn.Conv2d(160, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ELU(),
            # Additional conv layer for more complexity
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        
        # Flattened size: 192 * 2 * 4 = 1536
        self.flattened_size = 192 * 2 * 4
        
        # Deep path - Added one more layer
        self.fc_deep = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 768),  # Additional intermediate layer
            nn.BatchNorm1d(768),
            nn.ELU(),
            nn.Dropout(p=0.45),
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(p=0.3)
        )
        
        # Shortcut path
        self.fc_shortcut = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ELU()
        )
        
        # Final prediction layers - Added one more layer
        self.fc_final = nn.Sequential(
            nn.Linear(512, 256),  # Increased from 128 to 256
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(256, 128),  # Additional layer
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(p=0.15),
            
            nn.Linear(128, 64),
            nn.ELU(),
            
            nn.Linear(64, 32),
            nn.ELU(),
            
            nn.Linear(32, 1)
        )
        
        # Initialize Adam optimizer for this model
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def forward(self, input):
        x = input.permute(0, 3, 1, 2)
        
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = x.reshape(x.size(0), -1)
        
        deep_features = self.fc_deep(x)
        shortcut_features = self.fc_shortcut(x)
        
        combined = torch.cat([deep_features, shortcut_features], dim=1)
        
        output = self.fc_final(combined)
        
        return output

    def configure_optimizers(self):
        """Return optimizer and scheduler"""
        return self.optimizer, self.scheduler

    def update_lr(self, validation_loss):
        """Update learning rate based on validation loss"""
        self.scheduler.step(validation_loss)