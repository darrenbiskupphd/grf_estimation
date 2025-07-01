import numpy as np
import ezc3d
import mujoco
import mujoco.viewer
import time
import os
from utils import rollout_centroid_positions

def load_marker_data(c3d_path):
    """
    Load marker positions from C3D file.
    
    Args:
        c3d_path: Path to the C3D file
        
    Returns:
        marker_positions: Array of shape (framecount, 99, 3) - marker positions over time
    """
    c3d = ezc3d.c3d(c3d_path)
    
    # Extract points data: shape (4, 99, framecount)
    points = c3d['data']['points']
    marker_positions = points[:3, :, :]  # (3, 99, framecount) ignore the 4th dimension (all 1's)
    
    # Transpose to get (framecount, 99, 3)
    marker_positions = marker_positions.transpose(2, 1, 0)
    
    return marker_positions / 1000  # Convert from mm to meters


def load_grf_data(npy_path):
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


def setup_mujoco_scene():
    """
    Create a MuJoCo scene with a floor and lighting.
    
    Returns:
        model: MuJoCo model
        data: MuJoCo data
    """
    # Create a simple XML scene with floor and lighting
    xml_string = """
    <mujoco model="visualization">
        <worldbody>
            <light directional="false" pos="0 0 7" dir="0 0 -1" castshadow="true"/>
            <geom name="floor" size="10 10 .125" type="plane"/>
        </worldbody>W
        
        <visual>
            <rgba haze=".15 .25 .35 1"/>
        </visual>
    </mujoco>
    """
    
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    return model, data


def draw_markers(viewer, marker_positions):
    """
    Draw markers as red spheres in the MuJoCo viewer.
    """
    for marker_idx, pos in enumerate(marker_positions):
        # Add a new geometry to the scene if we have space
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            # Get the next available geometry slot
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            
            color = np.array([1.0, 0.0, 0.0, 1.0])  # Default Red color
            # draw feet markers in orange
            if (marker_idx >= 86 and marker_idx < 96) or (marker_idx >= 61 and marker_idx < 71):  # right foot markers and left foot markers
               color = np.array([1.0, 0.647, 0.0, 1.0])  # Orange
            
            # Initialize the sphere
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.01, 0, 0]),  # Sphere radius
                pos=pos,
                mat=np.identity(3).flatten(),
                rgba=color
            )
            
            # Increment the geometry count
            viewer.user_scn.ngeom += 1
    
    return len(marker_positions) # Return the number of markers drawn


def draw_grfs(viewer, cop, grf, scale=.003):
    """
    Draw ground reaction forces as cyan arrows in the MuJoCo viewer.
    """
    for i in range(2):  # Two force plates
        start = cop[i]  # Center of pressure
        direction = grf[i] / np.linalg.norm(grf[i])

        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, direction) # compute the rotation to align the Z-axis with the force direction, then put into quat vector

        # Convert the quaternion to a rotation matrix.
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)

        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        # Set arrow from CoP to CoP + force vector
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.01, 0.01, scale * np.linalg.norm(grf[i])]), 
            pos=start,
            mat=mat, # Will be overridden by mju_quat2Mat
            rgba=np.array([0.0, 1.0, 1.0, 1.0])  # Cyan color
        )
        
        # Increment the geometry count
        viewer.user_scn.ngeom += 1

def draw_centroid_prediction(viewer, centroid_rollout):
    for com in centroid_rollout:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.02, 0, 0]),  # Sphere radius
                    pos=com,
                    mat=np.identity(3).flatten(),
                    rgba=np.array([0.0, 0.5, 1.0, 1.0])  # Blue-ish color
                )
        viewer.user_scn.ngeom += 1

def play_animation(marker_positions, cop, grf, frame_rate=250):
    """
    Play animation of markers and ground reaction forces.
    
    Args:
        marker_positions: Array of marker positions (framecount, 99, 3)
        cop: Center of pressure data (framecount, 2, 3)
        grf: Ground reaction force data (framecount, 2, 3)
        frame_rate: Animation frame rate (default 250 Hz)
    """
    # Setup MuJoCo scene
    model, data = setup_mujoco_scene()
    
    # Calculate time step
    dt = 1.0 / frame_rate
    
    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        max_frames = marker_positions.shape[0]

        # Configure camera position and orientation
        viewer.cam.lookat[0] = 0.0  # x position to look at
        viewer.cam.lookat[1] = 0.0  # y position to look at
        viewer.cam.lookat[2] = 0.0  # z position to look at (floor level)
        viewer.cam.distance = 5.0   # Distance from lookat point
        viewer.cam.elevation = -20  # Camera elevation angle (degrees)
        viewer.cam.azimuth = 45     # Camera azimuth angle (degrees)
        
        prev_com = np.mean(marker_positions[frame][5:13], axis=0) # torso markers
        while viewer.is_running():
            step_start = time.time()
            
            # Clear previous geometries
            viewer.user_scn.ngeom = 0

            com = np.mean(marker_positions[frame][5:13], axis=0) # torso markers
            com_vel = (com - prev_com) / dt  # Compute velocity

            # Draw current frame
            draw_markers(viewer, marker_positions[frame])
            draw_grfs(viewer, cop[frame], grf[frame])


            centroid_rollout = rollout_centroid_positions(mass=69.86, 
                                                          com_pos=com, 
                                                          com_vel=com_vel, 
                                                          cops=cop[frame], 
                                                          grfs=grf[frame], 
                                                          frame_rate=frame_rate, 
                                                          horizon=100)
            
            draw_centroid_prediction(viewer, centroid_rollout)

            # Update viewer
            mujoco.mj_step(model, data)
            viewer.sync()
            prev_com = com

            # Advance frame, ensures loop
            frame = (frame + 1) % max_frames
            
            # Rudimentary time keeping, will drift relative to wall clock
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    """
    Main function to load data and start visualization.
    """
    # Example file paths (modify as needed)
    c3d_path = 'data/GroundLink_dataset/mocap/s001_20220513/s001_20220513_ballethighleg_1.c3d'
    grf_path = 'data/GroundLink_dataset/force/s001_force/s001/s001_20220513_ballethighleg_1.npy'

    #c3d_path = 'data/GroundLink_dataset/mocap/s001_20220513/s001_20220513_chair_1.c3d'
    #grf_path = 'data/GroundLink_dataset/force/s001_force/s001/s001_20220513_chair_1.npy'

    #c3d_path = 'data/GroundLink_dataset/mocap/s005_20220610/s005_20220610_soccerkick_3_full.c3d'
    #grf_path = 'data/GroundLink_dataset/force/s005_force/s005/s005_soccerkick_full/s005_20220610_soccerkick_3.npy'
    
    marker_positions = load_marker_data(c3d_path)
    print(f"Loaded marker data with shape: {marker_positions.shape}")
    
    cop, grf = load_grf_data(grf_path)
    print(f"Loaded CoP data with shape: {cop.shape}")
    print(f"Loaded GRF data with shape: {grf.shape}")

    # Start animation
    play_animation(marker_positions, cop, grf * 9.81 * 69.86) # Scale GRF by 69.86 to de-normalize subject weight


if __name__ == "__main__":
    main()