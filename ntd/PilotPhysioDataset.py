import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
from ntd.utils.utils import standardize_array
from pathlib import Path

class CogPilotDataset(Dataset):
    """
    

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data_root, manifest, window_size=512, target_src=128):
        """
        Args:
            data_root (str): Path to PhysioNet dataset.
            manifest (str): Path to the expert label json
            window_size: 512 samples, 4 seconds @ 128 Hz
                seq_len (int): Number of timepoints per window (e.g., 8 seconds at 128hz).
            target_sr (int): 128 Hz
        """
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self.window_size = window_size
        self.target_sr = target_sr
        
        # load the Novice/Expert map
        #with open(label_map_path, 'r') as f:
        #    self.labels = json.load(f)
            
        #self.samples = self._index_dataset()
        #print(f"Dataset initialized with {len(self.samples)} valid runs.")
    def __len__(self, idx):
        return len(self.array)
        
    def _load_single_modality(self, files_info):
        """"
        Loads a single CSV, checks the header file to see if we have a valid sample rate, it not. Downsample
        """
        
        continue

    
    
    
    
    """
    def __init__(self, base_dir, modalities, window_size=2000, step_size=1000):
        self.samples = []
        self.modalities = modalities
        
        #print("base dir: ", base_dir)
        #subject_folders = [f for f in os.listdir(base_dir) if os.path.join(base_dir, f)]
        #print("subject folders: ", subject_folders)

        # Loop through each subject folder
        for subj_dir in sorted(os.listdir(base_dir)):
            subj_path = os.path.join(base_dir, subj_dir)
            if not os.path.isdir(subj_path):
                continue

            # Then session folder
            for ses_dir in os.listdir(subj_path):
                ses_path = os.path.join(subj_path, ses_dir)
                if not os.path.isdir(ses_path):
                    continue
                
                
                for level_dir in os.listdir(ses_path):
                    level_path = os.path.join(ses_path, level_dir)
                    
                    if not os.path.isdir(level_path):
                        continue

                    # Loop over modalities
                    for modality in modalities:
                        pattern = os.path.join(
                            level_path, f"*{modality}*_hea.csv"
                        )
                        for file in glob.glob(pattern):
                            df = pd.read_csv(file)
                            # Replace this with your signal column
                            signal = df.iloc[:, 1].to_numpy()  # e.g., 'data' or 'value'

                            # Window the signal
                            for start in range(0, len(signal) - window_size, step_size):
                                window = signal[start:start + window_size]
                                self.samples.append({
                                    "data": window,
                                    "modality": modality,
                                    "subject": subj_dir,
                                    "session": ses_dir,
                                    "level": level_dir
                                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.tensor(sample["data"], dtype=torch.float32)
        return data, sample["modality"]
    
    def __init__(
        self, 
        base_dir=None,
        modalities=None,
        signal_length=2000,
        step_size=None,
        filepath=None,
    ):
        super().__init__()
        
        # Handle both base_dir and filepath arguments (for compatibility)
        if filepath is not None:
            base_dir = filepath
        if base_dir is None:
            raise ValueError("Must provide base_dir or filepath")
        
        self.signal_length = signal_length
        self.step_size = step_size if step_size is not None else signal_length
        self.modalities = modalities
        
        print(f"Initializing PhysioDataset...")
        print(f"  Base directory: {base_dir}")
        print(f"  Modalities: {modalities}")
        print(f"  Signal length: {signal_length}")
        print(f"  Step size: {self.step_size}")
        
        # Collect all data using rglob (recursive glob)
        data_root = Path(base_dir)
        collected_signals = []
        file_count = 0
        
        for csv_path in data_root.rglob("*.csv"):
            name = csv_path.name.lower()
            
            # Check if this file matches any modality
            matched_modality = None
            for mod in modalities:
                if f"stream-{mod}" in name:
                    matched_modality = mod
                    break
            
            # Skip if not one of the physiological streams we want
            if matched_modality is None:
                continue
            
            file_count += 1
            
            # Extract metadata from path
            parts = csv_path.parts
            subject = next((p for p in parts if p.startswith("sub-")), "unknown")
            session = next((p for p in parts if p.startswith("ses-")), "unknown")
            level_run = next((p for p in parts if "level-" in p and "run-" in p), "unknown")
            
            # Load CSV
            try:
                df = pd.read_csv(csv_path)
                
                # Get signal column - typically the second column (index 1)
                # Adjust if your data structure is different
                if df.shape[1] < 2:
                    print(f"  Warning: {csv_path.name} has fewer than 2 columns, skipping")
                    continue
                
                # Assume signal is in column 1 (adjust if needed)
                signal = df.iloc[:, 1].to_numpy().astype(np.float32)
                
                # Check signal length
                if len(signal) < signal_length:
                    print(f"  Warning: {csv_path.name} too short ({len(signal)} < {signal_length}), skipping")
                    continue
                
                # Window the signal with sliding window
                for start in range(0, len(signal) - signal_length + 1, self.step_size):
                    window = signal[start:start + signal_length]
                    
                    # Check for NaN or Inf
                    if np.isnan(window).any() or np.isinf(window).any():
                        continue
                    
                    collected_signals.append({
                        "data": window,
                        "modality": matched_modality,
                        "subject": subject,
                        "session": session,
                        "level_run": level_run,
                        "source_file": str(csv_path.name)
                    })
                    
            except Exception as e:
                print(f"  Error loading {csv_path.name}: {e}")
                continue
        
        print(f"\n  Processed {file_count} CSV files")
        print(f"  Created {len(collected_signals)} signal windows")
        
        if len(collected_signals) == 0:
            raise ValueError(
                f"No samples were created! Checked {file_count} files.\n"
                "Check:\n"
                "1. Modality names match 'stream-{modality}' pattern in filenames\n"
                "2. CSV files have at least 2 columns\n"
                "3. Signals are long enough (>= signal_length)\n"
                f"4. Files exist in {base_dir}"
            )
        
        # Store metadata
        self.samples = collected_signals
        
        # Convert to array and standardize (following reference datasets)
        print(f"\n  Standardizing data...")
        all_windows = np.array([s["data"] for s in self.samples])  # Shape: (N, signal_length)
        
        # Add channel dimension to match reference format: (N, num_channels, signal_length)
        all_windows = all_windows[:, np.newaxis, :]  # Shape: (N, 1, signal_length)
        
        # Standardize across samples and time (axis=(0, 2))
        self.array, self.arr_mean, self.arr_std = standardize_array(
            all_windows, ax=(0, 2), return_mean_std=True
        )
        
        self.num_channels = 1  # Single channel per modality
        
        print(f"  Data mean: {self.arr_mean.item():.4f}")
        print(f"  Data std: {self.arr_std.item():.4f}")
        print(f"  Final array shape: {self.array.shape}")
        print(f"\n✓ Dataset initialization complete!")

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        # Follow the reference dataset format
        return_dict = {}
        
        # Return signal as torch tensor with shape (num_channels, signal_length)
        return_dict["signal"] = torch.from_numpy(np.float32(self.array[index]))
        
        # Optionally add conditioning (placeholder for now)
        cond = self.get_cond()
        if cond is not None:
            return_dict["cond"] = cond
        
        # Optionally add metadata
        # return_dict["modality"] = self.samples[index]["modality"]
        # return_dict["subject"] = self.samples[index]["subject"]
        
        return return_dict

    def get_cond(self):
        #Placeholder for conditional information.
        #Can be extended to return modality conditioning, etc.
  
        return None

def load_physio_data(
    base_dir,
    modalities,
    signal_length=2000,
    step_size=None,
):

    Convenience function to load physiological data.
    Mimics the structure of load_ajile_data() from reference.
    
    Args:
        base_dir: Path to data directory
        modalities: List of modality names
        signal_length: Window length
        step_size: Stride for sliding window
    
    Returns:
        dataset: PhysioDataset instance

    dataset = PhysioDataset(
        base_dir=base_dir,
        modalities=modalities,
        signal_length=signal_length,
        step_size=step_size,
    )
    return dataset


def test_dataset():
    base_dir = "../dataPackage/task-ils"
    modalities = [
        'lslshimmereda', 
        'lslshimmerecg',
        'lslshimmeremg',
        'lslshimmerresp',
        'lslshimmertorsoacc'
    ]
    
    print("=" * 70)
    print("Testing PhysioDataset")
    print("=" * 70)
    
    try:
        dataset = PhysioDataset(
            base_dir=base_dir,
            modalities=modalities,
            signal_length=2000,
            step_size=1000,
        )
        
        print("\n" + "=" * 70)
        print("Dataset Test Results")
        print("=" * 70)
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n✓ First sample:")
            print(f"  Keys: {sample.keys()}")
            print(f"  Signal shape: {sample['signal'].shape}")
            print(f"  Signal dtype: {sample['signal'].dtype}")
            print(f"  Signal range: [{sample['signal'].min():.3f}, {sample['signal'].max():.3f}]")
            
            # Check a few more samples
            print(f"\n✓ Sample shapes consistent:")
            for i in range(min(5, len(dataset))):
                s = dataset[i]
                print(f"  Sample {i}: {s['signal'].shape}")
        
        # Show dataset coverage
        print(f"\n✓ Dataset coverage:")
        subjects = set(s['subject'] for s in dataset.samples)
        sessions = set(s['session'] for s in dataset.samples)
        modalities_found = set(s['modality'] for s in dataset.samples)
        
        print(f"  Unique subjects: {len(subjects)}")
        print(f"  Unique sessions: {len(sessions)}")
        print(f"  Modalities: {sorted(modalities_found)}")
        
        # Samples per modality
        print(f"\n✓ Samples per modality:")
        for mod in sorted(modalities_found):
            count = sum(1 for s in dataset.samples if s['modality'] == mod)
            print(f"  {mod}: {count}")
        
        return dataset
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_dataset()"""
