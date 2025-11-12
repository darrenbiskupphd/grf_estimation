import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from utils import * 
import time as _time


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
    for i in range(cop.shape[0]):  # Iterate over each force plate
        start = cop[i]  # Center of pressure
        if np.linalg.norm(grf[i]) < 1e-6:  # Skip if force is negligible
            direction = np.array([0.0, 0.0, -1.0])  # Default direction if no force
        else:
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

def draw_markers_from_list(viewer, marker_list, color=np.array([0.0, 0.5, 1.0, 1.0]), size=0.02):
    for com in marker_list:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([size, 0, 0]),  # Sphere radius
                    pos=com,
                    mat=np.identity(3).flatten(),
                    rgba=color
                )
        viewer.user_scn.ngeom += 1

def play_animation(marker_positions, cop, grf, subject_mass=70, frame_rate=100):
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
        

        while viewer.is_running():
            step_start = time.time()
            # Clear previous geometries
            viewer.user_scn.ngeom = 0

            # Draw current frame
            draw_markers(viewer, marker_positions[frame])
            draw_grfs(viewer, cop[frame], grf[frame])

            # Update viewer
            mujoco.mj_step(model, data)
            viewer.sync()

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
    b3d_path = "data/AddBiomechanicsDataset/train/With_Arm/Han2023_Formatted_With_Arm/s006_split1/s006_split1.b3d"
    cop, grf, marker_clouds, subject_mass, freq = load_data_b3d(b3d_path,trial_num=5) ## sometimes each recording has multiple "trials"
        
    # Start animation
    play_animation(marker_clouds, cop, grf, subject_mass=subject_mass, frame_rate=freq)

if __name__ == "__main__":
    main()