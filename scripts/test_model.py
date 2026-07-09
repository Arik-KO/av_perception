"""Test that the model runs a forward pass with our data."""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from av_perception.data.dataset import CARLABEVDataset
from av_perception.models.simple_bev import SimpleBEVModel
from av_perception.utils.logger import setup_logger

setup_logger()

# Load dataset
dataset = CARLABEVDataset(
    data_dir=Path("data/raw/town10"),
    image_size=(224, 224),
    use_bev_target=True,
    bev_size=(200, 200),
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Create model
model = SimpleBEVModel(
    num_cameras=6,
    bev_height=200,
    bev_width=200,
    bev_channels=64,
    output_channels=3,
)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Device: {device}")

# Test forward pass
batch = next(iter(dataloader))
cameras = batch["cameras"].to(device)
bev_target = batch["bev_target"].to(device)

print(f"\nInput:  cameras shape = {cameras.shape}")
print(f"Target: bev_target shape = {bev_target.shape}")

# Forward pass
with torch.no_grad():
    bev_pred = model(cameras)

print(f"Output: bev_pred shape = {bev_pred.shape}")
print(f"Output range: [{bev_pred.min():.3f}, {bev_pred.max():.3f}]")

# Quick sanity check — compute loss
loss_fn = torch.nn.MSELoss()
loss = loss_fn(bev_pred, bev_target)
print(f"\nMSE Loss (untrained): {loss.item():.4f}")
print("Forward pass successful!")
