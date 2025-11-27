import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        residual = self.res_conv(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class UNetEnhanced(nn.Module):
    def __init__(self, in_channels=2, base_channels=128, dropout=0.1):
        super().__init__()
        # Encodeur
        self.enc1 = ResidualConvBlock(in_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels*2)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResidualConvBlock(base_channels*2, base_channels*4)
        self.drop = nn.Dropout2d(dropout)
        
        # Decodeur
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = ResidualConvBlock(base_channels*4, base_channels*2)
        
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = ResidualConvBlock(base_channels*2, base_channels)
        
        # Sortie
        self.out_conv = nn.Conv2d(base_channels, 1, 1)
    
    def forward(self, x, t):
        t_map = t.view(-1,1,1,1).expand(-1,1,x.size(2),x.size(3))
        x = torch.cat([x, t_map], dim=1)
        
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Bottleneck
        b = self.drop(self.bottleneck(self.pool(e2)))
        
        # Decodeur avec interpolation
        up2_b = F.interpolate(self.up2(b), size=e2.shape[2:], mode="nearest")
        d2 = self.dec2(torch.cat([up2_b, e2], dim=1))
        
        up1_d2 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode="nearest")
        d1 = self.dec1(torch.cat([up1_d2, e1], dim=1))
        
        out = self.out_conv(d1)
        return out
