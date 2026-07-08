"""Test that the Dataset loads our CARLA data correctly."""

from pathlib import Path
from torch.utils.data import DataLoader
from av_perception.data.dataset import CARLABEVDataset
from av_perception.utils.logger import setup_logger

setup_logger()

dataset = CARLABEVDataset(
    data_dir=Path("data/raw/town10"),
    image_size=(224, 224),
    use_bev_target=True,
    bev_size=(200, 200),
)

print(f"Dataset length: {len(dataset)}")

# Load one sample
sample = dataset[0]
print(f"\nSample contents:")
for key, value in sample.items():
    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

# Test DataLoader (batched loading)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(dataloader))
print(f"\nBatch contents:")
for key, value in batch.items():
    print(f"  {key}: shape={value.shape}")
