import numpy as np
import h5py
import os
from dataclasses import dataclass, asdict
import time
import traceback
import matplotlib.pyplot as plt

# --- Part 1: Parameter Setup (Unchanged) ---
@dataclass
class SimParameters2D:
    gamma: float = 0.05
    g_EM: float = 0.1
    omega: float = 50.0
    kappa: float = 0.05
    eta: float = 0.01
    Lx: float = 100.0; Ly: float = 100.0
    t_final: float = 1.0
    Nx: int = 64; Ny: int = 64
    Nt: int = 2000
    save_interval: int = 40
    rho_0: float = 1e5; c_scaled: float = 300.0
    m_phi_sq: float = 0.01; B0: float = 0.1
    rho_dissipation: float = 0.05

    def get_params_dict(self):
        p = asdict(self)
        p['dx'] = p['Lx']/p['Nx']; p['dy'] = p['Ly']/p['Ny']
        p['dt'] = p['t_final']/p['Nt']
        return p

# --- Part 2: Numerical Solver (Unchanged) ---
def laplacian_2d(f, dx, dy):
    return ((np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dx**2 +
           (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dy**2)

def gradient_2d(f, dx, dy):
    grad_x = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    grad_y = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dy)
    return grad_x, grad_y

# --- Part 3: Main Execution Block (Corrected Solver Loop) ---
def run_solver(p, f):
    print("--- Starting RUFT 2D Solver Loop (v3.1 - Final Data Build) ---")

    # (Grid & Initial Conditions setup is the same)
    x = np.linspace(0, p['Lx'], p['Nx'], endpoint=False)
    y = np.linspace(0, p['Ly'], p['Ny'], endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U = np.zeros((3, p['Nx'], p['Ny'])); phi = np.zeros((3, p['Nx'], p['Ny']))
    Bz = np.zeros((3, p['Nx'], p['Ny']))
    vx = np.zeros((p['Nx'], p['Ny'])); vy = np.zeros((p['Nx'], p['Ny']))
    rho = 0.1 * p['rho_0'] * np.exp(-((X - p['Lx']/2)**2 + (Y - p['Ly']/2)**2) / (p['Lx']/8)**2)
    Bz[1, ...] = p['B0']

    core_mask = np.sqrt((X - p['Lx']/2)**2 + (Y - p['Ly']/2)**2) < p['Lx']/4
    tau_E_history, time_history = [], []

    # HDF5 setup
    param_group = f.create_group("SIMULATION_PARAMETERS"); param_group.attrs.update(p)
    mesh_group = f.create_group("MESH_DATA"); mesh_group.create_dataset("x_grid", data=x); mesh_group.create_dataset("y_grid", data=y)
    timesteps_group = f.create_group("TIMESTEPS")

    start_time = time.time()
    for n in range(1, p['Nt'] - 1):
        prev, curr, next = 0, 1, 2

        # --- Physics calculations (unchanged) ---
        J_x, J_y = gradient_2d(Bz[curr, ...], p['dx'], p['dy']); J_y, J_x = J_y, -J_x
        F_x = J_y * Bz[curr, ...]; F_y = -J_x * Bz[curr, ...]
        vx += p['dt'] * F_x / rho; vy += p['dt'] * F_y / rho

        grad_rho_x, grad_rho_y = np.gradient(rho, p['dx'], p['dy'])
        S_rho = p['kappa'] * (grad_rho_x**2 + grad_rho_y**2)
        F_munu_Fmunu = -2 * Bz[curr, ...]**2
        phi_source = S_rho - p['g_EM'] * F_munu_Fmunu

        phi_lap = laplacian_2d(phi[curr, ...], p['dx'], p['dy'])
        phi_dot = (phi[curr, ...] - phi[prev, ...]) / p['dt']

        phi[next, ...] = (2*phi[curr, ...] - phi[prev, ...] + p['dt']**2 * (
            p['c_scaled']**2 * phi_lap - p['m_phi_sq'] * phi[curr, ...] + phi_source
        ) - p['dt'] * p['gamma'] * phi_dot)

        IP = p['rho_0'] * phi[curr, ...]
        U_lap = laplacian_2d(U[curr, ...], p['dx'], p['dy'])
        U[next, ...] = (2*U[curr, ...] - U[prev, ...] + p['dt']**2 * (p['c_scaled']**2 * U_lap + p['kappa'] * IP))

        E_field_x = -(vy * Bz[curr, ...]); E_field_y = (vx * Bz[curr, ...])
        curl_E_z = (gradient_2d(E_field_y, p['dx'], p['dy'])[0] - gradient_2d(E_field_x, p['dx'], p['dy'])[1])
        Bz_lap = laplacian_2d(Bz[curr, ...], p['dx'], p['dy'])
        Bz_dot = -curl_E_z + p['eta'] * Bz_lap
        Bz[next, ...] = Bz[curr, ...] + p['dt'] * Bz_dot

        rho -= p['dt'] * p['rho_dissipation'] * rho

        U[next, ...]=np.clip(U[next, ...],-1e12,1e12)
        phi[next, ...]=np.clip(phi[next, ...],-1e6,1e6)
        Bz[next, ...]=np.clip(Bz[next, ...], -10*p['B0'], 10*p['B0'])
        rho[:]=np.clip(rho,0,1e7)

        U[0],U[1]=U[1],U[2]; phi[0],phi[1]=phi[1],phi[2]; Bz[0],Bz[1]=Bz[1],Bz[2]

        if (n + 1) % p['save_interval'] == 0:
            current_time_s = (n + 1) * p['dt']
            time_history.append(current_time_s)

            # --- THE CRITICAL FIX: Save the full field data at each interval ---
            step_group = timesteps_group.create_group(str(n + 1))
            fields_group = step_group.create_group("fields")
            # Save the 'current' state, which is the most recently calculated one
            fields_group.create_dataset("U", data=U[1, ...], compression="gzip")
            # We save 'U' as it's the most direct representation of energy density for the GW calculation

            stats_group = step_group.create_group("stats")
            stats_group.attrs["sim_time_s"] = current_time_s
            # --- END OF FIX ---

            # Diagnostics can still be calculated and printed
            E_core = np.sum(U[1, ...][core_mask]) * p['dx'] * p['dy']
            P_loss = np.sum(p['rho_dissipation'] * U[1, ...]) * p['dx'] * p['dy']
            tau_E = np.abs(E_core / P_loss) if P_loss > 1e-12 else np.inf
            tau_E_history.append(tau_E)

            print(f"Step: {n+1}/{p['Nt']}, Time: {current_time_s:.3f}s, tau_E: {tau_E:.2e}s, Max U: {np.max(U[1, ...]):.2e}")

    # Save history data
    f.create_dataset("tau_E_history", data=np.array(tau_E_history))
    f.create_dataset("time_history", data=np.array(time_history))
    return time_history, tau_E_history

def main():
    print("--- RUFT 2D Fusion Simulation (v3.1 - Final Data Build) ---")
    params = SimParameters2D()
    unitless_params = params.get_params_dict()
    run_id = f"v3.1_g{params.gamma}_w{int(params.omega)}_gEM{params.g_EM}"
    # (File handling and plotting logic as before, now generating v3.1 files)
    output_filename = f"output_{run_id}.h5"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    try:
        with h5py.File(output_path, 'w') as f:
            time_hist, tau_E_hist = run_solver(unitless_params, f)
        print(f"--- Simulation Finished. Data saved to {output_path} ---")
        plt.figure(figsize=(10, 6))
        plt.plot(time_hist, tau_E_hist); plt.xlabel("Time (s)"); plt.ylabel("Confinement Time tau_E (s)")
        plt.title(f"RUFT v5.0 (Reactor Core v3.1): {run_id}"); plt.grid(True); plt.yscale('log')
        plot_path = os.path.join(output_dir, f"tau_E_plot_{run_id}.png")
        plt.savefig(plot_path)
        print(f"Confinement plot saved to {plot_path}")
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred."); traceback.print_exc()

if __name__ == "__main__":
    main()
