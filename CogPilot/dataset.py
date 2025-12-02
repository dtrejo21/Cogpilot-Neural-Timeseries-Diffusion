import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
from ntd.utils.utils import standardize_array
from pathlib import Path
from .preprocessing import CogPilotPhysiologicalDataSynchronizer
import pickle
import json
import logging
log = logging.getLogger(__name__)

class CogPilotDataset(Dataset):
    """
    Dataset of of the CogPilot Physio data
            
    Returns:
        Array of shape(batch, channels, time)
        
    - conditional on expert/novince label
    - 14 channels: ECG(3) + EDA(2) + EMG(5) + Resp(1) + ACC(3)
    - 4 second windows: 512 time points at 128 Hz
    """
    def __init__(self, index_json, signal_length=512, target_fs=128.0, split='train'):
        super().__init__()
        
        self.signal_length = signal_length
        self.target_fs = target_fs
        self.stats_file = "cogpilot_global_stats.pkl"
        self.run_lengths_file = "data/processed/cogpilot_run_lengths.pkl"
            
        self.run_index = self._load_and_split_index(index_json, split)
        
        # Synchronizer
        self.synch = CogPilotPhysiologicalDataSynchronizer(target_fs=self.target_fs)
        
        self.run_lengths = {}
            
        # compute normalization stats
        if split == 'train':
            print("Calculating stats & indexing lengths from training data...")
            
            self.channel_mean, self.channel_std, self.run_lengths = self._calculate_global_stats()
            
            # Save stats for test
            with open(self.stats_file, 'wb') as f:
                pickle.dump({'mean': self.channel_mean, 'std': self.channel_std}, f)
                
            # save run lengths
            with open(self.run_lengths_file, 'wb') as f:
                pickle.dump(self.run_lengths, f)
            
        else:
            # load stats computed from training
            if not os.path.exists(self.stats_file):
                print("Stats file not found")
                
            print(f"Loading global stats from {self.stats_file}")
            with open(self.stats_file, 'rb') as f:
                stats = pickle.load(f)
                
            self.channel_mean = stats['mean']
            self.channel_std = stats['std']
        
        # build windowed sample index
        self.samples = []
        self._build_sample_index()
        
        """ run_index_len = len(self.run_index)
        print(f"CogPilotDataset ({split}): {len(self.samples)} windows from {len(self.run_index)} runs")\
        
        # Count expert vs novice
        n_expert = sum(1 for s in self.samples if s['is_expert'] == 1)
        print(f"  Expert windows: {n_expert}, Novice windows: {len(self.samples) - n_expert}") """
        
    def _calculate_global_stats(self):
        #Calculate the mean and std, saves length of each run to self.run_lengths  
        #Returns
           # - mean & std
        run_lengths = {}
        sum_x = np.zeros(14)
        sum_sq_x = np.zeros(14)
        total_count = 0
        
        n = 0
        M = np.zeros(14)  # Running mean
        S = np.zeros(14)  # Running sum of squared differences from mean
        
        print(f"Scanning {len(self.run_index)} runs...")
        
        for i, run_info in enumerate(self.run_index):
            # synchronize the run
            #try:
            df_synch = self.synch.synchronize_run(run_info['files'])
            
            # save run length
            run_lengths[i] = len(df_synch)
            
            numeric_cols = df_synch.select_dtypes(include=[np.number]).columns
            vals = df_synch[numeric_cols].values  # Shape: (n_samples, 14)
            
            
            # Welford's algorithm: update mean and variance incrementally
            for sample in vals:
                n += 1
                delta = sample - M
                M += delta / n
                delta2 = sample - M
                S += delta * delta2          
                    
            """ # Accumulate Stats
            sum_x += np.sum(vals, axis=0) # sum across time
            sum_sq_x += np.sum(vals ** 2, axis=0) # sum of squares
            total_count += len(df_synch) """
                
            """except Exception as e:
                pass
                #print(f"Skipping run {i}: {e}") """
                # 
        
        #print(f"\nStats done. Processed {total_count} timepoints.")

        # compute mean and std
        """ mean_val = sum_x / total_count
        var_val = (sum_sq_x / total_count) - (mean_val ** 2)
        std_val = np.sqrt(np.maximum(var_val, 0))
        print(f"mean val: {mean_val}") """
        
        #print(f"\nStatistics computed from {n} total samples across {len(run_lengths)} runs")
        
        # Compute final mean and std
        mean_val = M
        variance_val = S / (n - 1) if n > 1 else np.zeros(14)
        std_val = np.sqrt(variance_val)
        
        # Add small epsilon to prevent division by zero
        std_val = np.maximum(std_val, 1e-8)

        """ print(f"\nFinal Statistics:")
        print(f"  Channel means: min={mean_val.min():.3f}, max={mean_val.max():.3f}, mean={mean_val.mean():.3f}")
        print(f"  Channel stds:  min={std_val.min():.3f}, max={std_val.max():.3f}, mean={std_val.mean():.3f}") """
        
        # Print per-channel stats
        channel_names = [
            'ECG_LL_RA', 'ECG_LA_RA', 'ECG_VX_RL',
            'PPG', 'EDA',
            'ACC_FA_X', 'ACC_FA_Y', 'ACC_FA_Z', 'EMG_FLEX', 'EMG_EXT',
            'RESP',
            'ACC_T_X', 'ACC_T_Y', 'ACC_T_Z'
        ]
        print(f"\nPer-channel statistics:")
        for idx, name in enumerate(channel_names):
            print(f"  {name:12s}: mean = {mean_val[idx]:8.3f}, std = {std_val[idx]:8.3f}")
        
        # Convert to torch tensors with shape (14, 1) for broadcasting
        mean_tensor = torch.from_numpy(mean_val).float().unsqueeze(1)
        std_tensor = torch.from_numpy(std_val).float().unsqueeze(1)
        
        return mean_tensor, std_tensor, run_lengths
            
    def _load_and_split_index(self, index_json, split):
        # Load index
        with open(index_json, "r") as f:
            index_json = json.load(f)
            
        # split data by runs
        n_train = int(len(index_json) * 0.8)
       
        if split == 'train': # First 80%
            return index_json[:n_train]
        else: # test (Remaining 20%)
            return index_json[n_train:]
        
    def _build_sample_index(self):
        """
        Build index of all valid windows across all runs
        Each entry in self.samples represents one 4-second window
        """
        
        """ n_samples = 0
        for run_indx, run_info in enumerate(self.run_index):
            df_sync = self.synch.synchronize_run(run_info['files'])
            
            
            n_samples = len(df_sync)
            print(f"n samples: {n_samples}")
            
            # Calculate number of windows for this run
            n_windows = (n_samples - self.signal_length) // 2
            
            # Create entry for each window
            for window_idx in range(n_windows):
                start_idx = window_idx
                
                self.samples.append({
                    'run_idx': run_indx,
                    'start_idx': start_idx,
                    'is_expert': run_info['is_expert'],
                    'subject': run_info['subject'],
                    'subject_num': run_info['subject_num'],
                    'run_id': run_info['run_id']
                })  """
        
        n_samples = 0
        for run_indx, run_info in enumerate(self.run_index):
            
            if run_indx in self.run_lengths:
                n_samples = self.run_lengths[run_indx]
            else:
                # We are in test set
                df_sync = self.synch.synchronize_run(run_info['files'])
                n_samples = len(df_sync)
            
            # Calculate number of windows for this run
            n_windows = (n_samples - self.signal_length) // 2
            
            # Create entry for each window
            for window_idx in range(n_windows):
                start_idx = window_idx
                
                self.samples.append({
                    'run_idx': run_indx,
                    'start_idx': start_idx,
                    'is_expert': run_info['is_expert'],
                    'subject': run_info['subject'],
                    'subject_num': run_info['subject_num'],
                    'run_id': run_info['run_id']
                })
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, indx):
        """
        Get one window of data
        
        Returns:
        
            - signal: (channels, time) = (14, 512) tensor
            - cond: (1,) dict with expert/novince label
            - label: (1,) tensor for classification experiments
        """
        sample_info = self.samples[indx]
        run_info = self.run_index[sample_info['run_idx']]
        
        # Load a single run and stack all modalities together
        df_synch = self.synch.synchronize_run(run_info['files'])
        
        # Extract window
        start = sample_info['start_idx']
        end = start + self.signal_length
        
        # Safety check for time
        if end > len(df_synch):
            start = max(0, len(df_synch) - self.signal_length)
            end = len(df_synch)
        
        # Get numpy array: (time, channels) - (512, 14)
        window = df_synch.iloc[start:end].values
        
        # Ensure correct length
        if len(window) < self.signal_length:
            pad_length = self.signal_length - len(window)
            window = np.pad(window, ((0, pad_length), (0, 0)), mode='edge')
            
            
        # Transpose to (Channels, Time) -> (14, 512)
        x = torch.from_numpy(window.T).float()
        
        """
        # BELOW WILL HAVE THE BIGGEST CHANGE
        # Normalize 
        # Calculate Mean and Std for EACH channel separately (axis=1 is time)
        # keepdims=True ensures shapes are (14, 1) for broadcasting
        ch_mean = np.mean(x, axis=1, keepdims=True)
        ch_std = np.std(x, axis=1, keepdims=True)
        
        # Add Epsilon to prevent division by zero
        # If a channel is flat (std=0), this makes the divisor 1e-8
        epsilon = 1e-8
        x_norm = (x - ch_mean) / (ch_std + epsilon) """
        # Normalize channels
        x_norm = (x - self.channel_mean) / self.channel_std
        
        # Convert to Float Tensor
        #signal_tensor = torch.from_numpy(x_norm).float()
        is_expert = float(sample_info['is_expert'])
        cond_tensor = torch.ones(1, self.signal_length) * is_expert
        #cond_tensor = torch.tensor([is_expert], dtype=torch.float32)  # Shape: (1,) <- doesn't work for recent changes
        label_tensor = torch.tensor([is_expert], dtype=torch.float32)
        
        
        return {
            "signal": x_norm, 
            "cond": cond_tensor,  #.float(),
            "label":  label_tensor
        }
        
    def denomoralize(self, normalized_data):
        """
        Denormalize data back to its original scale

        Args:
            normalized_data (torch.Tensor of shape): (n_channels, n_timepoints)
            
        Returns:
            denormalized_data: same shape as input
        """
        if normalized_data.ndim == 2:
            # (channels, time) case
            return normalized_data * self.channel_std + self.channel_mean
       
            