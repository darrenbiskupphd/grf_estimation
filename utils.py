import numpy as np

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
    total_grf = grfs.sum(dim=0)
    total_force = np.tile(total_grf + mass * g, (horizon, 1))
    acc = total_force / mass  # (horizon, 3)
    # Integrate velocity 
    vel_steps = com_vel + np.cumsum(acc, axis=0) * dt
    # Integrate position
    pos_steps = com_pos + np.cumsum(vel_steps, axis=0) * dt
    return [pos_steps[i] for i in range(horizon)]
