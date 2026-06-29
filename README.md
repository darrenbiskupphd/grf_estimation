This repository is dedicated to the research and implementation of Ground Reaction Force and Center of Pressure estimation algorithms using MuJoCo physics simulation.

## Environment Modifications

This repository utilizes a modified version of the standard Gymnasium MuJoCo `humanoid.xml`.

```xml
<default class="foot">
    <!-- condim="6" enables torsional/rolling friction computation. 
            Friction array: sliding, torsional, rolling -->
    <geom size=".027" condim="6" friction="1.0 0.05 0.0001"/>
    <default class="foot1">
        <geom fromto="-.07 -.01 0 .14 -.03 0"/>
    </default>
    <default class="foot2">
        <geom fromto="-.07 .01 0 .14  .03 0"/>
    </default>
    </default>
```

### Physical Rationale
1. **Analytic Collision Geometry:** Replacing the rigid spheres (or boxes) with capsules prevents point-contact simulation. This yields continuous, differentiable contact normals required for stable spatial wrench extraction.
2. **6D Contact Mechanics:** The foot capsules are initialized with `condim="6"`. This directs the MuJoCo solver to compute the full 6-DOF contact wrench at the geometries, capturing the 3D linear friction components as well as the torsional (vertical Z-torque) and rolling friction components. 
3. **Biomechanical Footprint:** The heel and toe capsules approximate the spatial bounding box of a human foot, allowing for proper internal lever arms when resolving the individual contact wrenches into a singular resultant CoP per foot.

## Anthropometric Baseline Calibration & The Geometric Inertia Heuristic

### The Problem
To train a generalized Ground Reaction Force (GRF) regressor, the simulation pipeline must generate thousands of randomized human morphologies. We want to randomize over individual segment geometries, which will produce varying, uncorrelated inertia characteristics so that the trained regressor generalizes to diverse human physiques and limb proportions. 

### The Approach
We use MuJoCo's `inertiafromgeom="true"` compiler flag to let the simulator natively compute segment inertia from a uniform density. By assuming a heuristic density roughly equivalent to water ($\rho = 1000 \text{ kg/m}^3$), we establish a direct mathematical link between a segment's spatial volume and its inertial tensor.

We independently mutate the geometry (length $L_i$ and radius $r_i$) of individual segments. This forces the neural network to learn the implicit mapping from a localized marker cloud volume to that specific limb's localized inertial contribution.

### Establishing the Nominal Baseline
To establish a mathematically grounded starting point before applying independent randomizations, we anchor our baseline humanoid to a nominal 50th percentile male ($M = 75 \text{ kg}$, $H = 1.75 \text{ m}$) using foundational biomechanical data (Winter, 2009).

For a given segment $i$, we extract its nominal mass $m_i$ and length $L_i$ using Winter's anthropometric scaling fractions:

$$m_i = c_{mass, i} M$$
$$L_i = c_{length, i} H$$

The exact volume $V$ of a MuJoCo capsule is the sum of a cylinder and a spherical cap ($V = \pi r^2 L + \frac{4}{3} \pi r^3$). Isolating the capsule radius ($r_i$), which maps directly to the `size` attribute in the MJCF XML:

### The Domain Randomization Strategy
During data generation, the simulation script applies independent uniform noise to the nominal $L_i$ and $r_i$ values derived above. MuJoCo's compiler then automatically resolves the updated localized mass and inertia tensor for that specific mutated segment on the fly.


Here is the explicit mapping of Winter's 50th percentile male (75kg, 1.75m) to the XML variables. This table directly reflects the values computed by the cubic root solver.

| Anatomical Segment | Winter Mass Fraction | Target Mass (m) | Winter Length Fraction | Target Length (L) | XML Target name | Imputed Size (r) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Head | 0.081 M | 6.07 kg | - | - | `head` | 0.113 m |
| Trunk (Total) | 0.497 M | 37.27 kg | 0.300 H | 0.525 m | - | - |
| Trunk (Per Capsule) | - | 9.32 kg | - | 0.140 m | `torso`, `waist_upper`, `waist_lower`, `butt`| 0.104 m |
| Thigh | 0.100 M | 7.50 kg | 0.245 H | 0.429 m | `thigh_left`, `thigh_right` | 0.068 m |
| Shin | 0.046 M | 3.49 kg | 0.246 H | 0.430 m | `shin_left`, `shin_right` | 0.048 m |
| Upper Arm | 0.028 M | 2.10 kg | 0.186 H | 0.325 m | `upper_arm_left`, `upper_arm_right` | 0.042 m |
| Forearm | 0.016 M | 1.20 kg | 0.146 H | 0.255 m | `lower_arm_left`, `lower_arm_right` | 0.036 m |
| Foot (Total) | 0.0145 M | 1.09 kg | 0.152 H | 0.266 m | `foot_*`, `toe_*` | 0.027 m |

Male Spinal Kinematic Offsets (pos Z-translation)

- Head: $0.207$ m
- Lumbar (waist_lower): $-0.283$ m
- Pelvis (pelvis): $-0.180$ m
- Hip Sockets (thigh): $-0.044$ m

Female Baseline Parameters (M = 60 kg, H = 1.63 m)

| Anatomical Segment | Mass Fraction | Target Mass (m) | Length Fraction | Target Length (L) | Imputed Size (r) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Head | 0.0820 M | 4.92 kg | - | - | 0.105 m |
| Trunk (Total) | 0.4684 M | 28.10 kg | 0.300 H | 0.489 m | - |
| Trunk (Per Capsule) | - | 7.02 kg | - | 0.140 m | 0.093 m |
| Thigh | 0.1175 M | 7.05 kg | 0.245 H | 0.399 m | 0.068 m |
| Shin | 0.0535 M | 3.21 kg | 0.246 H | 0.401 m | 0.047 m |
| Upper Arm | 0.0265 M | 1.59 kg | 0.186 H | 0.303 m | 0.038 m |
| Forearm | 0.0137 M | 0.82 kg | 0.146 H | 0.238 m | 0.031 m |
| Foot (Total) | 0.0145 M | 0.87 kg | 0.152 H | 0.248 m | 0.025 m |

Female Spinal Kinematic Offsets (pos Z-translation) Derived by scaling the nominal male offsets by the height ratio ($S = 1.63 / 1.75 = 0.9314$).

- Head: $0.195$ m
- Lumbar (waist_lower): $-0.267$ m
- Pelvis (pelvis): $-0.169$ m
- Hip Sockets (thigh): $-0.041$ m