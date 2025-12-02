import sys
import os
import torch
import numpy as np
import pprint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path if running from a subfolder
sys.path.append(".") 

from CogPilot.dataset import CogPilotDataset

CONFIG = {
    "index_json": "data/processed/dataset_index.json",
    "signal_length": 512,
    "target_fs": 128.0
}

# -----------------------------
# Verify Split Logic
# -----------------------------

print("\n===== VERIFYING SPLITS =====")

# Initialize both datasets
print("Loading Training Set...")
train_ds = CogPilotDataset(split="train", **CONFIG)

print("\nLoading Testing Set...")
test_ds = CogPilotDataset(split="test", **CONFIG)

# Extract Run IDs from the internal index
# run_index is a list of dicts: [{'run_id': '...', ...}, ...]
train_ids = set([f"{r['subject']}_{r['run_id']}" for r in train_ds.run_index])
test_ids  = set([f"{r['subject']}_{r['run_id']}" for r in test_ds.run_index])

n_total = len(train_ids) + len(test_ids)
print(f"Train Unique Recordings: {len(train_ids)}")
print(f"Test Unique Recordings:  {len(test_ids)}")
print(f"Total Recordings:        {n_total}")
print(f"Split Ratio:             {len(train_ids)/n_total:.1%} / {len(test_ids)/n_total:.1%}")

# 5. Check Overlap
intersection = train_ids.intersection(test_ids)

if len(intersection) == 0:
    print("\n✅ SUCCESS: No overlap between Training and Testing sets.")
else:
    print(f"\n❌ CRITICAL ERROR: Found {len(intersection)} recordings in BOTH sets!")
    print(f"Overlapping: {list(intersection)}...")
    
    
train_subjs = set([x.split('_')[0] for x in train_ids])
test_subjs = set([x.split('_')[0] for x in test_ids])

subj_overlap = train_subjs.intersection(test_subjs)

print("\n--- Subject Leakage Check ---")
if len(subj_overlap) == 0:
    print("✅ SUCCESS: Subjects are strictly separated.")
    print(f"Train Subjects: {len(train_subjs)}")
    print(f"Test Subjects:  {len(test_subjs)}")
else:
    print(f"⚠️ WARNING: {len(subj_overlap)} subjects appear in both sets (Split is by Run, not Subject).")
   
   
""" 
# check the global mean and std
means = dataset.channel_mean.squeeze()
stds = dataset.channel_std.squeeze()

print(f"Ch 0 (ECG) Mean: {means[0]:.4f} | Std: {stds[0]:.4f}")
print(f"Ch 1 (ECG) Mean: {means[1]:.4f} | Std: {stds[1]:.4f}")
print(f"Ch 2 (ECG) Mean: {means[2]:.4f} | Std: {stds[2]:.4f}  <-- Should be ~338")
print(f"Ch 3 (EDA) Mean: {means[3]:.4f} | Std: {stds[3]:.4f}")

if means[2] > 100:
    print("\n✅ SUCCESS: Global Mean captures the physical offset.")
else:
    print("\n❌ FAILURE: Global Mean is too small. Did normalization fail?")"""


# -----------------------------
# Test __getitem__
# -----------------------------
print("\n===== GETITEM TEST =====")
sample = train_ds[0]  # Returns a dictionary now

# Extract components
signal = sample["signal"]
cond = sample["cond"]
label = sample["label"]

print(f"Signal shape: {signal.shape}") # Expect: [14, 512]
print(f"Cond shape:   {cond.shape}")   # Expect: [1, 512]
print(f"Label:        {label.item()}") # Expect: 0.0 or 1.0

# -----------------------------
# Test DataLoader
# -----------------------------
""" print("\n===== DATALOADER TEST =====")
loader = DataLoader(train_ds, batch_size=4, shuffle=True)

batch = next(iter(loader))

print(f"Batch keys: {batch.keys()}")             # Expect: ['signal', 'cond', 'label']
print(f"Batch Signal shape: {batch['signal'].shape}") # Expect: [4, 14, 512]
print(f"Batch Cond shape:   {batch['cond'].shape}")   # Expect: [4, 1, 512] """

# -----------------------------
#  Plot a raw window
# -----------------------------
""" plot_example = True

if plot_example:
    plt.figure(figsize=(12, 5))
    plt.title(f"Example Window (Channel 0) - Label: {label.item()}")
    
    # Use the signal we extracted earlier
    plt.plot(signal[0].numpy()) 
    
    plt.xlabel("Time points")
    plt.ylabel("Amplitude (Standardized)")
    plt.grid(True, alpha=0.3)
    plt.show()
 """