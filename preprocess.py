# %%
import os
import glob
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader

# %%
class VenusVPCMDataset(Dataset):
    def __init__(self, nc_files, wind_stats=None, so2_stats=None):
        """
        Custom PyTorch Dataset for Venus Planetary Climate Model (VPCM) NetCDF outputs.
        
        Args:
            nc_files (list): List of file paths to the VPCM .nc files.
            wind_stats (dict, optional): Mean and std of winds from the training set.
            so2_stats (dict, optional): Min and max of SO2 from the training set.
        """
        print(f"Loading {len(nc_files)} VPCM files into memory...")
        
        # 1. Load and concatenate all NetCDF files along the time dimension
        # open_mfdataset is highly optimized for this.
        self.ds = xr.open_mfdataset(nc_files, combine='nested', concat_dim='time')
        
        # Load into RAM for massively faster PyTorch __getitem__ fetching.
        # (A 96x96 grid will easily fit in memory on your 128-core nodes).
        self.ds.load() 
        
        # ---------------------------------------------------------
        # 2. EXTRACT VARIABLES (Replace with your exact VPCM keys)
        # Assuming shape becomes: (time, lat, lon) -> (N, 96, 96)
        # ---------------------------------------------------------
        self.u_winds = self.ds['u'].values  # Zonal wind
        self.v_winds = self.ds['v'].values  # Meridional wind
        
        # Extract SO2. If your SO2 is 3D (time, alt, lat, lon), you will need 
        # to slice it at your target altitude here (e.g., self.ds['so2'].isel(alt=target_idx))
        raw_so2_grid = self.ds['so2'].values 
        
        # ---------------------------------------------------------
        # 3. CALCULATE DISK-INTEGRATED SO2
        # ---------------------------------------------------------
        self.so2_targets = self._calculate_disk_integrated_so2(raw_so2_grid, self.ds['lat'].values)
        
        # ---------------------------------------------------------
        # 4. NORMALIZATION (Crucial for Neural Networks)
        # ---------------------------------------------------------
        self.wind_stats = wind_stats
        self.so2_stats = so2_stats
        self._normalize_data()

    def _calculate_disk_integrated_so2(self, so2_grid, latitudes):
        """
        Integrates the 2D SO2 map into a single scalar value per time step.
        Accounts for the spherical geometry by weighting by the cosine of latitude.
        """
        # Convert latitudes to radians for cosine weighting
        lat_rads = np.deg2rad(latitudes)
        cos_lats = np.cos(lat_rads)
        
        # Reshape cos_lats to broadcast across the (time, lat, lon) array
        # cos_lats shape: (96,) -> (1, 96, 1)
        cos_lats_3d = cos_lats[np.newaxis, :, np.newaxis]
        
        # Calculate the weighted average across lat (axis 1) and lon (axis 2)
        weighted_so2 = np.sum(so2_grid * cos_lats_3d, axis=(1, 2)) / np.sum(cos_lats_3d * np.ones_like(so2_grid), axis=(1, 2))
        
        # Reshape to (N, 1) to match PyTorch's expected target shape
        return weighted_so2.reshape(-1, 1)

    def _normalize_data(self):
        """Standardizes winds (Z-score) and scales SO2 to [0, 1]."""
        # If stats aren't provided (i.e., this is the Training set), calculate them.
        if self.wind_stats is None:
            self.wind_stats = {
                'u_mean': self.u_winds.mean(), 'u_std': self.u_winds.std(),
                'v_mean': self.v_winds.mean(), 'v_std': self.v_winds.std()
            }
        
        if self.so2_stats is None:
            self.so2_stats = {
                'min': self.so2_targets.min(), 'max': self.so2_targets.max()
            }
            
        # Apply Z-score normalization to winds
        self.u_winds = (self.u_winds - self.wind_stats['u_mean']) / self.wind_stats['u_std']
        self.v_winds = (self.v_winds - self.wind_stats['v_mean']) / self.wind_stats['v_std']
        
        # Apply Min-Max scaling to SO2 (helps the Gaussian NLL loss stay stable)
        so2_range = self.so2_stats['max'] - self.so2_stats['min']
        # Prevent division by zero if SO2 is perfectly constant
        so2_range = so2_range if so2_range > 0 else 1.0 
        self.so2_targets = (self.so2_targets - self.so2_stats['min']) / so2_range

    def __len__(self):
        # The number of samples is the number of time steps
        return len(self.u_winds)

    def __getitem__(self, idx):
        # Fetch the u and v grids for this specific time step
        u = self.u_winds[idx]
        v = self.v_winds[idx]
        
        # Stack them into a single tensor of shape (2, 96, 96)
        # We use float32 because it's the standard for PyTorch weights
        wind_tensor = torch.tensor(np.stack([u, v], axis=0), dtype=torch.float32)
        
        # Fetch the corresponding integrated SO2 scalar
        so2_tensor = torch.tensor(self.so2_targets[idx], dtype=torch.float32)
        
        return wind_tensor, so2_tensor