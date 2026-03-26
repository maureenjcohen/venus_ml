# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class VenusProbabilisticCNN(nn.Module):
    def __init__(self):
        super(VenusProbabilisticCNN, self).__init__()
        
        # -------------------------------------------------------------------
        # 1. FEATURE EXTRACTION (The "Physics Engine")
        # Note: We set padding=0 here because we handle the custom 
        # planetary padding manually in the forward pass.
        # -------------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size:
        # 96x96 -> pool1 -> 48x48 -> pool2 -> 24x24 -> pool3 -> 12x12
        # 64 channels * 12 * 12 = 9216
        self.flattened_size = 64 * 12 * 12
        
        # -------------------------------------------------------------------
        # 2. THE DENSE LAYERS (The "Calculator")
        # -------------------------------------------------------------------
        # Shared dense representation
        self.fc_shared = nn.Linear(in_features=self.flattened_size, out_features=256)
        
        # --- Branch A: Predicts the SO2 Mean (mu) ---
        self.fc_mu1 = nn.Linear(in_features=256, out_features=32)
        self.fc_mu_out = nn.Linear(in_features=32, out_features=1)
        
        # --- Branch B: Predicts the Uncertainty (sigma) ---
        self.fc_sigma1 = nn.Linear(in_features=256, out_features=32)
        self.fc_sigma_out = nn.Linear(in_features=32, out_features=1)

    def planetary_pad(self, x):
        """
        Applies circular padding to longitude (left/right) and 
        constant replication padding to latitude (top/bottom).
        Padding format for F.pad is (left, right, top, bottom).
        """
        # 1. Pad longitude (dim 3) circularly by 1 pixel on each side
        x = F.pad(x, pad=(1, 1, 0, 0), mode='circular')
        # 2. Pad latitude (dim 2) by replicating the pole pixels by 1
        x = F.pad(x, pad=(0, 0, 1, 1), mode='replicate')
        return x

    def forward(self, x):
        # Apply convolutions with custom padding
        x = self.pool(F.relu(self.conv1(self.planetary_pad(x))))
        x = self.pool(F.relu(self.conv2(self.planetary_pad(x))))
        x = self.pool(F.relu(self.conv3(self.planetary_pad(x))))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Shared dense layer
        x = F.relu(self.fc_shared(x))
        
        # --- Mu (Prediction) Branch ---
        mu = F.relu(self.fc_mu1(x))
        mu = self.fc_mu_out(mu) # Linear output for the continuous value
        
        # --- Sigma (Uncertainty) Branch ---
        sigma = F.relu(self.fc_sigma1(x))
        sigma = self.fc_sigma_out(sigma)
        # Apply Softplus to ensure standard deviation is strictly positive.
        # Uncertainty cannot be zero or negative!
        sigma = F.softplus(sigma) + 1e-6 
        
        return mu, sigma

# %%
class VenusWindToSO2CNN(nn.Module):
    def __init__(self):
        super(VenusWindToSO2CNN, self).__init__()
        
        # -------------------------------------------------------------------
        # ENCODER (Feature Extraction)
        # Learns the spatial patterns and dynamics of the u and v wind fields.
        # Input channels = 2 (u, v).
        # -------------------------------------------------------------------
        self.enc_conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling to reduce spatial dimensions and extract higher-level features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # -------------------------------------------------------------------
        # DECODER (Spatial Reconstruction)
        # Upsamples the learned features back into a 2-D spatial map of SO2.
        # -------------------------------------------------------------------
        self.dec_upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        self.dec_upconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        
        # Final layer maps the features to a single channel (SO2 concentration)
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        # --- Encoder Pass ---
        # Shape: (Batch, 2, Lat, Lon)
        e1 = F.relu(self.enc_conv1(x))
        p1 = self.pool(e1)
        
        # Shape: (Batch, 16, Lat/2, Lon/2)
        e2 = F.relu(self.enc_conv2(p1))
        p2 = self.pool(e2)
        
        # Shape: (Batch, 32, Lat/4, Lon/4)
        e3 = F.relu(self.enc_conv3(p2))
        
        # --- Decoder Pass ---
        # Upsample and apply convolutions
        d1 = F.relu(self.dec_upconv1(e3))
        d1 = F.relu(self.dec_conv1(d1))
        
        d2 = F.relu(self.dec_upconv2(d1))
        d2 = F.relu(self.dec_conv2(d2))
        
        # Output Shape: (Batch, 1, Lat, Lon)
        # Assuming SO2 is scaled/normalized, a linear activation (or ReLU if strictly > 0) 
        # is appropriate for the final regression output.
        out = self.final_conv(d2)
        
        return out
