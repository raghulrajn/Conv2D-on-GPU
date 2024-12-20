import torch
import numpy as np
import os

# Load the model checkpoint
weights = torch.load("unet_carvana_scale0.5_epoch2.pth", 
                     weights_only=True,
                     map_location=torch.device('cpu'))


output_dir = "kernels"
os.makedirs(output_dir, exist_ok=True)

for name, param in weights.items():
    if "weight" in name:
        print(f"Extracting: {name} | Shape: {param.shape}")
        
        kernel = param.cpu().numpy()

        filename = os.path.join(output_dir, f"{name.replace('.', '_')}.npy")
        np.save(filename, kernel)
        print(f"Saved {name} to {filename}")

print("All kernels have been saved.")
