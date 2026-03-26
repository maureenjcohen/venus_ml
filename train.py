# %%
# Import packages
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Local imports
from models import VenusProbabilisticCNN

# %%
if __name__ == "__main__":

    log_filename = "training_log_v1_baseline.csv"
    
    # Write the header only if the file doesn't exist yet
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])

    # Force PyTorch to use multiple CPU cores (adjust based on your node)
    torch.set_num_threads(8) 

    model = VenusProbabilisticCNN()
    criterion = nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # ==========================================
    # --- DUMMY DATALOADERS ---
    # We generate fake 96x96 global wind maps and fake SO2 scalars.
    # ==========================================
    print("Generating dummy data...")
    num_train_samples = 128
    num_val_samples = 32
    batch_size = 16
    
    # Training Data (128 samples)
    dummy_train_winds = torch.randn(num_train_samples, 2, 96, 96) 
    dummy_train_so2 = torch.rand(num_train_samples, 1) # Random SO2 between 0.0 and 1.0
    train_dataset = TensorDataset(dummy_train_winds, dummy_train_so2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation Data (32 samples)
    dummy_val_winds = torch.randn(num_val_samples, 2, 96, 96)
    dummy_val_so2 = torch.rand(num_val_samples, 1)
    val_dataset = TensorDataset(dummy_val_winds, dummy_val_so2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("Starting training loop...")

    #=================================================
    # ---- REAL DATA EVENTUALLY ---
    # =================================================
    # 1. Define your file splits
    # all_vpcm_files = glob.glob("/path/to/your/vpcm/outputs/*.nc")
    # train_files = all_vpcm_files[:7] # e.g., 7 simulations for training
    # val_files = all_vpcm_files[7:]   # e.g., 3 simulations for validation

    # # 2. Initialize the Training Dataset
    # train_dataset = VenusVPCMDataset(train_files)

    # # 3. Extract the scaling stats from the Training set!
    # # This is critical: the validation set MUST be normalized using the 
    # # training set's statistics to simulate looking at unseen data.
    # w_stats = train_dataset.wind_stats
    # s_stats = train_dataset.so2_stats

    # # 4. Initialize the Validation Dataset using the Training stats
    # val_dataset = VenusVPCMDataset(val_files, wind_stats=w_stats, so2_stats=s_stats)

    # # 5. Create the DataLoaders
    # # Set num_workers > 0 to use your 128 cores for background data loading
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Train
    epochs = 100
    for epoch in range(epochs):
    
        # ==========================================
        # 1. TRAINING PHASE
        # ==========================================
        model.train()  # Tells PyTorch to enable Dropout and track gradients
        running_train_loss = 0.0
        
        for wind_inputs, true_so2 in train_loader: 
            optimizer.zero_grad()
            
            # Forward pass
            pred_mu, pred_sigma = model(wind_inputs)
            pred_variance = pred_sigma ** 2
            
            # Calculate Training Loss
            loss = criterion(pred_mu, true_so2, pred_variance)
            
            # BACKWARD PASS: The model updates its knowledge
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        # Calculate the average training loss for this epoch
        avg_train_loss = running_train_loss / len(train_loader)

        # ==========================================
        # 2. VALIDATION PHASE
        # ==========================================
        model.eval()  # Tells PyTorch to act deterministically
        running_val_loss = 0.0
        
        with torch.no_grad(): 
            for wind_inputs, true_so2 in val_loader: 
                
                # Forward pass
                pred_mu, pred_sigma = model(wind_inputs)
                pred_variance = pred_sigma ** 2
                
                # Calculate Validation Loss
                loss = criterion(pred_mu, true_so2, pred_variance)
                
                running_val_loss += loss.item()
                
        # Calculate the average validation loss for this epoch
        avg_val_loss = running_val_loss / len(val_loader)
        
        # ==========================================
        # 3. LOGGING
        # ==========================================
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Append metrics to the CSV file safely
        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])