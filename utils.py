import numpy as np
import ezc3d
import nimblephysics as nimble
import pandas as pd
from scipy.signal import butter, filtfilt


def rollout_centroid_positions(
    mass: float,
    com_pos: np.ndarray,
    com_vel: np.ndarray,
    cops: np.ndarray,
    grfs: np.ndarray,
    frame_rate: float,
    horizon: int,
) -> list:
    """
    Vectorized rollout of future centroid positions using Euler integration.
    """
    dt = 1.0 / frame_rate
    g = np.array([0, 0, -9.81])
    # Repeat GRFs for horizon steps (assume constant over horizon)
    #print(f"GRFs shape: {grfs.shape}")
    total_grf = grfs.sum(axis=0)
    total_force = np.tile(total_grf + mass * g, (horizon, 1))
    acc = total_force / mass  # (horizon, 3)
    # Integrate velocity 
    vel_steps = com_vel + np.cumsum(acc, axis=0) * dt
    # Integrate position
    pos_steps = com_pos + np.cumsum(vel_steps, axis=0) * dt
    return [pos_steps[i] for i in range(horizon)]


def load_marker_data(c3d_path):
    """
    Load marker positions from C3D file.
    
    Args:
        c3d_path: Path to the C3D file
        
    Returns:
        marker_positions: Array of shape (framecount, num_markers, 3) - marker positions over time
    """
    c3d = ezc3d.c3d(c3d_path)
    
    # Extract points data: shape (4, num_markers, framecount)
    points = c3d['data']['points']
    marker_positions = points[:3, :, :]  # (3, num_markers, framecount) ignore the 4th dimension (all 1's)

    # Transpose to get (framecount, num_markers, 3)
    marker_positions = marker_positions.transpose(2, 1, 0)
    
    return marker_positions / 1000  # Convert from mm to meters


def load_grf_data_npy(npy_path):
    """
    Load ground reaction force data from NPY file.
    
    Args:
        npy_path: Path to the NPY file containing force data
        
    Returns:
        cop: Center of pressure data, shape (framecount, 2, 3)
        grf: Ground reaction force data, shape (framecount, 2, 3)
    """
    force_data = np.load(npy_path, allow_pickle=True).item()
    
    cop = force_data['CoP']  # (framecount, 2, 3)
    grf = force_data['GRF']  # (framecount, 2, 3)
    
    return cop, grf

def load_data_jeonghan(csv_path):
    """
    Load ground reaction force data from the first block of a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        tuple: (cop, grf) arrays each with shape [frames, 2 plates, 3 coordinates]
    """
    # Read the file skipping the first 5 rows (header rows)
    yoga_data = pd.read_csv(csv_path, skiprows=5)
    
    # Extract the first trial/block based on the value in the second column
    nan_row_idx = yoga_data.index[yoga_data.iloc[:, 1:].isna().all(axis=1)]
    grf_data = yoga_data.loc[:nan_row_idx[0]-1]
    
    # Extract the needed columns (12 columns total, 6 for each force plate)
    fp1_force = grf_data.iloc[:, 2:5].to_numpy()    # 3 columns for force plate 1 force
    fp1_cop = grf_data.iloc[:, 8:11].to_numpy()     # 3 columns for force plate 1 CoP
    fp2_force = grf_data.iloc[:, 11:14].to_numpy()  # 3 columns for force plate 2 force
    fp2_cop = grf_data.iloc[:, 17:20].to_numpy()    # 3 columns for force plate 2 CoP

    grf = np.stack([fp1_force, fp2_force], axis=1)
    cop = np.stack([fp1_cop, fp2_cop], axis=1) 

    # Find where the marker trajectory data startsgra
    traj_idx = yoga_data.index[yoga_data.iloc[:, 0] == "Trajectories"][0]
    marker_start_idx = traj_idx + 5  # Skip 5 rows after "Trajectories"
    
    # Extract marker data from this point on
    marker_data = yoga_data.iloc[marker_start_idx:]
    # Drop the first 2 columns of marker_data
    marker_clouds = marker_data.iloc[:, 2:]
    # Convert marker data to numpy array and reshape it to [frames, num_markers, 3]
    num_markers = marker_clouds.shape[1] // 3
    marker_clouds = marker_clouds.iloc[:, :num_markers*3].to_numpy().reshape(marker_clouds.shape[0], num_markers, 3)

    return cop.astype(np.float64)/1000, -1*grf.astype(np.float64), marker_clouds.astype(np.float64)/1000


def load_data_b3d(b3d_path, trial_num=0):
    """
    Load marker and ground reaction force data from a B3D file.
    
    Args:
        b3d_path: Path to the B3D file
    """
    # Load subject from B3D file
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Load frames from a trial
    trial = trial_num
    frames = subject.readFrames(
        trial=trial,
        startFrame=0,
        numFramesToRead=subject.getTrialLength(trial),
        includeProcessingPasses=True  # This is the correct parameter if you need processing passes
    )

    num_force_plates = subject.getNumForcePlates(trial=trial)
    cop = np.zeros((len(frames), num_force_plates, 3))  # Center of pressure
    grf = np.zeros((len(frames), num_force_plates, 3))  # Ground reaction forces
    num_markers = len(frames[0].markerObservations)
    marker_clouds = np.zeros((len(frames), num_markers, 3))
    for i, frame in enumerate(frames):
        cop[i] = frame.rawForcePlateCenterOfPressures
        grf[i] = frame.rawForcePlateForces
        observed_markers = [pos[1] for pos in frame.markerObservations]
        num_observed = len(observed_markers)
        if num_observed > 0:
            marker_clouds[i, :num_observed, :] = np.array(observed_markers)

    cop = cop[:, :, [0, 2, 1]]
    grf = grf[:, :, [0, 2, 1]]
    marker_clouds = marker_clouds[:, :, [0, 2, 1]]

    return cop, grf, marker_clouds, subject.getMassKg()

def lowpass_filter(data: np.ndarray, cutoff_freq: float, fs: float) -> np.ndarray:
    nyquist = 0.5 * fs
    norm_cutoff = cutoff_freq / nyquist
    b, a = butter(N=4, Wn=norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)