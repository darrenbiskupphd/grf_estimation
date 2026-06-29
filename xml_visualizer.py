# pyright: reportAttributeAccessIssue=false
import mujoco
import mujoco.viewer
import numpy as np
import sys

def calculate_capsule_radius(mass: float, length: float, density: float = 1000.0) -> float:
    """Solves the cubic volume equation to find the required capsule radius."""
    a = (4.0 / 3.0) * np.pi * density
    b = np.pi * density * length
    c = 0.0
    d = -mass
    
    roots = np.roots([a, b, c, d])
    real_roots = roots[(np.isreal(roots)) & (roots > 0)]
    return float(np.real(real_roots[0]))

def validate_masses(model: mujoco.MjModel):
    """Validates the inertial mass of all bodies."""
    print(f"\n{'Body Name':<20} | {'Mass (kg)'}")
    print("-" * 35)
    
    total_mass = 0.0
    for i in range(1, model.nbody):  # Skip worldbody (index 0)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = model.body_mass[i]
        total_mass += mass
        print(f"{name:<20} | {mass:.3f}")
        
    print("-" * 35)
    print(f"{'Total Mass':<20} | {total_mass:.3f} kg\n")

def validate_inertias(model: mujoco.MjModel):
    """Compares compiled spatial inertia against Plagenhoef empirical targets."""
    print(f"{'Limb Name':<18} | {'I_comp (kg*m^2)':<15} | {'I_Plagenhoef (kg*m^2)':<17} | {'Error (%)'}")
    print("-" * 65)

    # Plagenhoef transverse radius of gyration (k) fractions for females
    plagenhoef_k = {
        "thigh_right": 0.328, "thigh_left": 0.328,
        "shin_right": 0.293, "shin_left": 0.293,
        "upper_arm_right": 0.322, "upper_arm_left": 0.322,
        "lower_arm_right": 0.303, "lower_arm_left": 0.303
    }

    for name, k in plagenhoef_k.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        m = model.body_mass[body_id]
        
        # Total capsule length = cylinder length + 2*radius
        r = model.geom_size[geom_id][0]
        half_L_cyl = model.geom_size[geom_id][1]
        L = (2 * half_L_cyl) + (2 * r)

        i_plagenhoef = m * (k * L)**2
        
        # Transverse inertia is the maximum principal inertia for a long cylinder
        i_comp = max(model.body_inertia[body_id])
        
        err = abs(i_comp - i_plagenhoef) / i_plagenhoef * 100
        print(f"{name:<18} | {i_comp:<15.4f} | {i_plagenhoef:<17.4f} | {err:.1f}%")
    print("-" * 65 + "\n")

def validate_kinematics(model: mujoco.MjModel, data: mujoco.MjData):
    """Runs forward kinematics to validate total height and CoM."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    # Calculate highest point (Head)
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head")
    head_top = data.geom_xpos[head_id][2] + model.geom_size[head_id][0]

    # Calculate lowest point across all geoms
    min_z = float('inf')
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
            
        radius = model.geom_size[i][0]
        half_length = model.geom_size[i][1] if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
        z_axis = data.geom_xmat[i].reshape(3, 3)[:, 2]
        
        ep1_z = data.geom_xpos[i][2] + (z_axis[2] * half_length)
        ep2_z = data.geom_xpos[i][2] - (z_axis[2] * half_length)
        
        geom_bottom = min(ep1_z, ep2_z) - radius
        if geom_bottom < min_z:
            min_z = geom_bottom

    total_height = head_top - min_z
    com_z = data.subtree_com[1][2] 
    com_fraction = (com_z - min_z) / total_height

    print(f"{'Kinematic Parameter':<25} | {'Value'}")
    print("-" * 40)
    print(f"{'Total Height':<25} | {total_height:.3f} m")
    print(f"{'CoM Z-Height':<25} | {(com_z - min_z):.3f} m")
    print(f"{'CoM Fraction (% of H)':<25} | {com_fraction*100:.1f} %")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_xml.py <path_to_xml>")
        sys.exit(1)
        
    xml_path = sys.argv[1]
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    validate_masses(model)
    validate_inertias(model)
    validate_kinematics(model, data)
    
    mujoco.viewer.launch(model, data)