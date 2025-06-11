import numpy as np
import h5py
import os
from dataclasses import dataclass, asdict
import time
import traceback
import matplotlib.pyplot as plt

# --- Part 1: Parameter Setup (Unchanged from your working version) ---
@dataclass
class ReplicatorSimParametersV2:
    Lx: float = 10.0; Ly: float = 10.0; Lz: float = 10.0
    Nx: int = 16; Ny: int = 16; Nz: int = 16
    scan_time: float = 0.05; print_time: float = 0.05
    Nt_per_phase: int = 200
    save_interval: int = 20
    c_scaled: float = 300.0; kappa: float = 1.0; m_phi_sq: float = 0.1
    gamma: float = 0.1; g_EM: float = 1.0
    scan_zeta: float = 100.0; scan_delta: float = 200.0
    print_zeta: float = 200.0; print_delta: float = 50.0
    rho_seed_value: float = 10.0
    hologram_center: tuple = (5.0, 5.0, 5.0)
    hologram_radius: float = 2.0
    hologram_omega: float = 500.0

    def get_params_dict(self):
        p = asdict(self)
        p['dx'] = p['Lx']/p['Nx']; p['dy'] = p['Ly']/p['Ny']; p['dz'] = p['Lz']/p['Nz']
        p['dt'] = self.scan_time / self.Nt_per_phase
        return p

# --- Part 2: Numerical Solver Core (Unchanged) ---
def laplacian_3d(f, dx, dy, dz):
    return ((np.roll(f, -1, axis=2) - 2*f + np.roll(f, 1, axis=2)) / dz**2 +
           (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dy**2 +
           (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dx**2)

# --- Part 3: Main Simulation Engine (HDF5 Logic added) ---
def run_solver(p, h5_file):
    print("--- Starting RUFT 3D Replicator Solver Loop (v2.3) ---")

    # (Setup logic is identical)
    x = np.linspace(0, p['Lx'], p['Nx'], endpoint=False)
    y = np.linspace(0, p['Ly'], p['Ny'], endpoint=False)
    z = np.linspace(0, p['Lz'], p['Nz'], endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    U = np.zeros((3, p['Nx'], p['Ny'], p['Nz'])); phi = np.zeros((3, p['Nx'], p['Ny'], p['Nz']))
    rho = np.zeros((p['Nx'], p['Ny'], p['Nz']))
    pattern_buffer = np.zeros((p['Nt_per_phase'], p['Nx'], p['Ny'], p['Nz']))
    hologram_mask = np.sqrt((X - p['hologram_center'][0])**2 + (Y - p['hologram_center'][1])**2 + (Z - p['hologram_center'][2])**2) < p['hologram_radius']
    total_rho_history = []; time_history = []

    # --- FIX: HDF5 Group Setup within the solver ---
    timesteps_group = h5_file.create_group("TIMESTEPS")

    # PHASE 1: SCANNING
    print("\n--- BEGINNING PHASE 1: SCANNING ---")
    rho[hologram_mask] = p['rho_seed_value']
    for n in range(1, p['Nt_per_phase']):
        t_curr = n * p['dt']
        # (Physics is identical)
        hologram_field = np.zeros_like(X); hologram_field[hologram_mask] = np.sin(p['hologram_omega'] * t_curr)
        interaction_term = p['g_EM'] * phi[1, ...] * (-2 * hologram_field**2)
        rho_dot = p['scan_zeta'] * interaction_term * rho - p['scan_delta'] * rho
        rho += p['dt'] * rho_dot; rho[:]=np.clip(rho, 0, 1e6)
        pattern_buffer[n, ...] = np.abs(rho_dot)

        total_rho = np.sum(rho) * p['dx'] * p['dy'] * p['dz']
        total_rho_history.append(total_rho); time_history.append(t_curr)
        if (n + 1) % p['save_interval'] == 0:
            print(f"[Scan] Step: {n+1}/{p['Nt_per_phase']}, Time: {t_curr:.3f}s, Total Rho: {total_rho:.3e}")

    # PHASE 2: PRINTING
    print("\n--- BEGINNING PHASE 2: PRINTING ---")
    rho.fill(0)
    for n in range(1, p['Nt_per_phase']):
        t_curr = p['scan_time'] + (n * p['dt'])
        # (Physics is identical)
        pattern_playback = pattern_buffer[n, ...]
        rho_dot = p['print_zeta'] * pattern_playback - p['print_delta'] * rho
        rho += p['dt'] * rho_dot; rho[:]=np.clip(rho, 0, 1e6)

        total_rho = np.sum(rho) * p['dx'] * p['dy'] * p['dz']
        total_rho_history.append(total_rho); time_history.append(t_curr)
        if (n + 1) % p['save_interval'] == 0:
            # --- FIX: Saving data for the Print Phase ---
            step_group = timesteps_group.create_group(str(n + 1 + p['Nt_per_phase'])) # Unique key for this phase
            fields_group = step_group.create_group("fields")
            fields_group.create_dataset("rho_slice_xy", data=rho[:, :, p['Nz']//2], compression="gzip")
            stats_group = step_group.create_group("stats")
            stats_group.attrs["sim_time_s"] = t_curr
            print(f"[Print] Step: {n+1}/{p['Nt_per_phase']}, Time: {t_curr:.3f}s, Total Rho: {total_rho:.3e}")

    print(f"--- [Replicator Sim v2.3] Finished. ---")
    return rho, time_history, total_rho_history

def main():
    """Main execution function with robust I/O."""
    print("--- RUFT v5.0 Genesis Key: 3D Scan-and-Print Replicator (Final) ---")

    params = ReplicatorSimParametersV2()
    params_dict = params.get_params_dict()
    run_id = f"replicator_v2.3_Z{params.print_zeta}_d{params.print_delta}"

    sim_dir = "simulations"
    output_dir = os.path.join(sim_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{run_id}.h5")

    try:
        # --- FIX: Full HDF5 file handling is restored ---
        with h5py.File(output_path, 'w') as f:
            param_group = f.create_group("SIMULATION_PARAMETERS"); param_group.attrs.update(params_dict)
            mesh_group = f.create_group("MESH_DATA")
            mesh_group.create_dataset("x_grid", data=np.linspace(0, params_dict['Lx'], params_dict['Nx'], endpoint=False))
            mesh_group.create_dataset("y_grid", data=np.linspace(0, params_dict['Ly'], params_dict['Ny'], endpoint=False))
            mesh_group.create_dataset("z_grid", data=np.linspace(0, params_dict['Lz'], params_dict['Nz'], endpoint=False))

            final_rho, time_hist, rho_history = run_solver(params_dict, f)

            f.create_dataset("total_rho_history", data=np.array(rho_history))
            f.create_dataset("time_history", data=np.array(time_hist))

        print(f"--- Simulation Finished. Data successfully saved to {output_path} ---")

        # Plotting logic is now guaranteed to work with saved data
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Run ID: {run_id}", fontsize=16)

        final_rho_slice = final_rho[:, :, params.Nz//2]
        im = axes[0].imshow(final_rho_slice.T, origin='lower', extent=[0, params.Lx, 0, params.Ly])
        fig.colorbar(im, ax=axes[0], label="Final Mass Density (rho)")
        axes[0].set_title("Replicator Result: Final State XY Slice"); axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")

        axes[1].plot(time_hist, rho_history)
        axes[1].axvline(x=params_dict['scan_time'], color='r', linestyle='--', label='Scan/Print Transition')
        axes[1].set_title("History of Total Mass in Cavity"); axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Total Rho (Unitless)")
        axes[1].grid(True); axes[1].legend()

        plt.tight_layout(rect=[0,0,1,0.95])
        plot_path = os.path.join(output_dir, f"summary_plot_{run_id}.png")
        plt.savefig(plot_path)
        print(f"Summary plot saved to {plot_path}")
        plt.show()

    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred.")
        traceback.print_exc()

if __name__ == "__main__":
    main()
