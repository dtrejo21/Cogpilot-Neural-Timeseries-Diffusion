import hydra
import torch
import numpy as np
import pandas as pd
import pickle
import os
from omegaconf import OmegaConf
from ntd.train_diffusion_model import init_diffusion_model
from ntd.utils.kernels_and_diffusion_utils import generate_samples
from pathlib import Path
from datetime import datetime, timedelta
import logging
log = logging.getLogger(__name__)

CHANNEL_MAP = {
    'ecg':  slice(0, 3),
    'eda':  slice(3, 5),
    'emg':  slice(5, 10),
    'resp': slice(10, 11),
    'acc':  slice(11, 14)
}

FILE_SUFFIXES = {
    'ecg':  'stream-lslshimmerecg',
    'eda':  'stream-lslshimmereda',
    'emg':  'stream-lslshimmeremg',
    'resp': 'stream-lslshimmerresp',
    'acc':  'stream-lslshimmertorsoacc'
}

COL_NAMES = {
    'ecg':  ['ecg_projection_ll_ra_mV', 'ecg_projection_la_ra_mV', 'ecg_projection_vx_rl_mV'],
    'eda':  ['ppg_finger_mV', 'eda_hand_l_kOhms'],
    'emg':  ['accelerometry_forearm_r_x_mps2', 'accelerometry_forearm_r_y_mps2', 'accelerometry_forearm_r_z_mps2', 'emg_wrist_flexor_mV', 'emg_wrist_extensor_mV'],
    'resp': ['respiration_trace_mV'],
    'acc':  ['accelerometry_torso_x_mps2', 'accelerometry_torso_y_mps2', 'accelerometry_torso_z_mps2']
}

LEVEL_SEQUENCE = [
    "01B", "03B", "02B", "04B",  # Runs 1-4
    "03B", "04B", "01B", "02B",  # Runs 5-8
    "04B", "02B", "03B", "01B"   # Runs 9-12
]

OUTPUT_ROOT = Path("generated_data_root")
NUM_SUBJECTS = 20
WINDOWS_PER_RUN = 18
START_DT = datetime(2021, 1, 1, 12, 0, 0)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def generate(cfg):
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

    # Load trained model
    diffusion, network = init_diffusion_model(cfg)
    diffusion = diffusion.to(device)
    
    experiment_name = cfg.base.experiment  # debug_run
    model_filename = f"{experiment_name}_models.pkl"
    model_path = Path(experiment_name) / model_filename
    
    if os.path.exists(model_path):
        log.info(f"Loading model from {model_path}")
        state_dict = pickle.load(open(model_path, "rb"))
        diffusion.load_state_dict(state_dict)
    else:
        log.info("Warning: Model weights not found. Generating with random weights.")
    
    # Load global stats for unnormalization
    stats_path = 'cogpilot_global_stats.pkl'
    if os.path.exists(stats_path):
        log.info("Stats file found")
        
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            
            # Load mean/std and move to CPU numpy for calculation
            # Shape is likely (14, 1) or (14,)
            global_mean = stats['mean'].cpu().numpy().flatten()
            global_std = stats['std'].cpu().numpy().flatten()
            
            # Reshape for broadcasting: (14, 1) to match (Channels, Time)
            global_mean = global_mean[:, None]
            global_std = global_std[:, None]
    else:
        log.info("Stats file not found. Generated data will be standardized")
        global_mean = 0.0
        global_std = 1.0
        
        
    signal_length = cfg.dataset.signal_length
    fs = cfg.dataset.target_fs
    samples_per_run = WINDOWS_PER_RUN * signal_length
    
    # Generate Subjects
    for subject_index in range(1, NUM_SUBJECTS+1):
        current_dt = START_DT
        
        # conditioning, even IDs are experts, odds are novince
        is_expert = 1.0 if subject_index % 2 == 0 else 0.0
        subject_type = "Expert" if is_expert else "Novince"
        
        subject_id = f'sub-cp{subject_index:03d}'
        
        print(f"Generating {subject_id}")
        
        # Create Conditioning Tensor for this subject
        # Shape: (Batch, 1, Signal_Length) -> (4, 1, 512)  <- old version
        # shape: (batch, 1) for NTD AdaConv
        cond_tensor = torch.ones(WINDOWS_PER_RUN, 1, signal_length).to(device) * is_expert
        #cond_tensor = torch.ones(WINDOWS_PER_RUN, 1, signal).to(device) * is_expert

        #Session ID
        session_id = "ses-20230101"
        
        # Generate 4 Levels x 4 Runs 
        run_global_counter = 1
        levels = ["01B", "03B", "02B", "04B"]
        
        for i, level in enumerate(LEVEL_SEQUENCE):
            for run_num in range(1, 4): # 3 runs per level
                # Folder: level-01B_run-001
                run_num = i + 1
                
                run_name = f"level-{level}_run-{run_num:03d}"
                run_path = OUTPUT_ROOT / "task-ils" / subject_id / session_id / run_name
                run_path.mkdir(parents=True, exist_ok=True)
                
                # GENERATE SYNTHETIC DATA!!!!!
                with torch.no_grad():
                    samples = generate_samples(
                        diffusion=diffusion,
                        total_num_samples=WINDOWS_PER_RUN,
                        batch_size=WINDOWS_PER_RUN,
                        cond=cond_tensor
                    ) 
                    # samples shape: (Windows=4s, Channels=14, Time=512)
                    
                # Un-normalize data
                samples_np = samples.cpu().numpy()
                    
                # Stitch windows together to make one continuous "stream"
                # (4, 14, 512) -> (14, 4*512) -> (2048, 14) for CSV files
                stitched_signal_norm = samples.permute(1, 0, 2).reshape(14, -1).cpu().numpy() #.T
                
                # Apply the Inverse
                # global_std/mean is (14, 1)
                stitched_signal_raw = (stitched_signal_norm * global_std) + global_mean
                
                # Transpose to (Time, Channels) for CSV
                stitched_signal_final = stitched_signal_raw.T
                
                # Time column
                time_deltas = pd.to_timedelta(np.arange(samples_per_run) / fs, unit='s')
                time_col = current_dt + time_deltas
                current_dt = time_col[-1] + pd.Timedelta(seconds=5)
                
                ## Add time
                #n_samples = stitched_signal.shape[0]
                
                # SAVE INDIVIDUAL CSVs
                for mod, slicer in CHANNEL_MAP.items():
                    # Extract columns for this modality 
                    mod_data = stitched_signal_final[:, slicer]
                    
                    # Construct Filename
                    # sub-cp001_ses-20230101_task-ils_stream-lslshimmerecg_feat-chunk_level-01B_run-001_dat.csv
                    base_fname = f"{subject_id}_{session_id}_task-ils_{FILE_SUFFIXES[mod]}_feat-chunk_{run_name}"
                    dat_file = run_path / f"{base_fname}_dat.csv"
                    hea_file = run_path / f"{base_fname}_hea.csv"
                    
                    # Save Data CSV with headers
                    df = pd.DataFrame(mod_data, columns=COL_NAMES[mod])
                    df.insert(0, 'time_dn', time_col)
                    df.to_csv(dat_file, index=False)
                    
                    # Save Header CSV (Simple metadata)
                    with open(hea_file, 'w') as f:
                        f.write(f"samplingRate {cfg.dataset.target_fs}\n")
                        f.write(f"numChannels {mod_data.shape[1]}\n")
                        #f.write(f"is_expert {is_expert}\n")
                        
                    hea_df = pd.DataFrame({
                        #'time_dn': [time], # need to fix
                        'Fs_Hz': [fs],
                        #'sampleCount': [n_samples]
                    })
                    hea_df.to_csv(hea_file, index=False)

                run_global_counter += 1
            
    print("\nGeneration Complete!")
    print(f"Data saved to: {OUTPUT_ROOT.resolve()}")
  
if __name__ == "__main__":
    generate()
