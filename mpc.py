import scipy.sparse as sp
import osqp
import numpy as np

# Constants
g = 9.81
nx, nu = 7, 3  # state and control dimensions
horizon = 50

# Preallocated matrices (global)
# System dynamics matrices
Ac = np.zeros((nx, nx))
Bc = np.zeros((nx, nu))
Ad = np.zeros((nx, nx))
Bd = np.zeros((nx, nu))

# QP matrices
A_qp = np.zeros((nx*horizon, nx))
B_qp_data = np.zeros((horizon, horizon, nx, nu))  # Store dense data before creating sparse matrix
x0 = np.zeros(nx)
y_ref = np.zeros(horizon*nx)

# QP cost matrices (sparse)
Q_diag = np.array([0, 0, 100, 100, 100, 100, 1])  # weights on state error
R_diag = 0.01 * np.array([1, 1, 1])  # weights on control effort
# Terminal cost weights
Qf_diag = np.array([0, 0, 1000, 500, 500, 500, 1])  

# Control limits
u_min = np.array([-50.0, -50.0, -10.0])
u_max = np.array([50.0, 50.0, 0.0])  # Will be updated with mass*g
c_lower = np.tile(u_min, horizon)
c_upper = np.tile(u_max, horizon)

# Initialize sparse matrices once
Q = sp.diags(Q_diag)
R = sp.diags(R_diag)
L = sp.block_diag([Q] * (horizon - 1) + [sp.diags(Qf_diag)], format='csr')
K = sp.kron(sp.eye(horizon), R)
C = sp.eye(horizon * nu)

B_qp_lil = sp.lil_matrix((horizon*nx, horizon*nu))

def rollout_optimal_trajectory(
    mass: float,
    com_pos: np.ndarray,
    com_vel: np.ndarray,
    cops: np.ndarray,
    grfs: np.ndarray,
    frame_rate: float,
) -> np.ndarray:
    dt = 1.0 / frame_rate

    # Update continuous time dynamics matrix
    Ac[0, 3] = 1
    Ac[1, 4] = 1
    Ac[2, 5] = 1
    Ac[3, 6] = (grfs[0,0] + grfs[1,0]) / mass
    Ac[4, 6] = (grfs[0,1] + grfs[1,1]) / mass
    Ac[5, 6] = (grfs[0,2] + grfs[1,2]) / mass - g

    # Update continuous time input matrix
    Bc[3, 0] = 1 / mass
    Bc[4, 1] = 1 / mass
    Bc[5, 2] = 1 / mass

    # Discretize the A and B matrices
    np.multiply(dt, Bc, out=Bd)  # Bd = dt*Bc
    Ad[:] = np.eye(7)  # Set Ad to identity
    np.add(Ad, dt * Ac, out=Ad)  # Ad += dt*Ac
    
    # Build the full A matrix for the horizon
    A_qp[:nx, :] = Ad
    for i in range(1, horizon):
        np.matmul(Ad, A_qp[(i-1)*nx:i*nx, :], out=A_qp[i*nx:(i+1)*nx, :])
    
    # Build B_qp efficiently
    for row in range(horizon):
        for col in range(row + 1):
            A_power = A_qp[(row - col)*nx : (row - col + 1)*nx, :]
            B_block = A_power @ Bd
            r_start = row * nx
            c_start = col * nu
            B_qp_lil[r_start:r_start + nx, c_start:c_start + nu] = B_block
    B_qp = B_qp_lil.tocsr()  # Convert to CSR format once at the end

    # Update initial state
    x0[:3] = com_pos
    x0[3:6] = com_vel
    x0[6] = 1.0

    # Update reference trajectory (only the z-coordinate changes)
    y_ref.fill(0)  # Reset all values to zero
    y_ref[2::nx] = 2  # Set desired z-height (every nx elements starting from index 2)

    # Calculate QP matricesg
    A_qp_x0 = A_qp @ x0
    # H = 2*(B_qp.T @ L @ B_qp + K) - already sparse
    H = 2 * (B_qp.T @ L @ B_qp + K)
    G = 2 * B_qp.T @ L @ (A_qp_x0 - y_ref)
    
    # Update upper control limit based on mass
    c_upper[2::nu] = mass * g  # Update z-force upper limits

    # Solve quadratic program
    prob = osqp.OSQP()
    prob.setup(H, G, C, c_lower, c_upper, 
               alpha=1.0,
               eps_abs=1e-4,
               eps_rel=1e-4,
               max_iter=4000,
               adaptive_rho=True,
               verbose=False)
    
    # Solve problem
    res = prob.solve()
    U = res.x
    np.set_printoptions(suppress=True)
    print("optimal control solution:", np.round(U[:3], 4))

    # Calculate rollout without creating intermediate large matrices
    rollout = A_qp_x0 + B_qp @ U
    return rollout.reshape(horizon, nx)[:, :3]


