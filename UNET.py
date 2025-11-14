import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet4(nn.Module):
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        # Encodeur
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels*2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1)
        
        # Decodeur
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Conv2d(base_channels*2, base_channels, 3, padding=1)
        
        # Sortie
        self.out_conv = nn.Conv2d(base_channels, 1, 1)
    
    def forward(self, x, t):
        t_map = t.view(-1,1,1,1).expand(-1,1,x.size(2),x.size(3))
        x = torch.cat([x, t_map], dim=1)
        
        # Encodeur
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        
        # Bottleneck
        b = F.relu(self.bottleneck(self.pool(e2)))
        
        # Decodeur avec interpolation pour corriger les tailles
        up2_b = F.interpolate(self.up2(b), size=e2.shape[2:], mode="nearest")
        d2 = F.relu(self.dec2(torch.cat([up2_b, e2], dim=1)))
        
        up1_d2 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode="nearest")
        d1 = F.relu(self.dec1(torch.cat([up1_d2, e1], dim=1)))
        
        out = self.out_conv(d1)
        return out