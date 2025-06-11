import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- Configuration ---
# This script is designed to be run from the root of the "RUFT v4.0" project directory.
SIM_DATA_PATH = os.path.join("SIMULATIONS", "2D_Solver_Dedalus", "output_2d.h5")

# --- Data Loading Function ---
def load_data_2d(filepath):
    """Loads and returns data from the specified 2D HDF5 file."""
    print(f"--- Loading data from {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            # Load mesh data
            x_grid = f['/MESH_DATA/x_grid'][:]
            y_grid = f['/MESH_DATA/y_grid'][:]

            # Load parameters for context
            params = dict(f['/SIMULATION_PARAMETERS'].attrs)

            # Load timesteps
            timesteps = sorted(f['/TIMESTEPS'].keys(), key=int)
            print(f"Found {len(timesteps)} saved timesteps.")

            # Pre-allocate arrays for performance
            num_steps = len(timesteps)
            Nx, Ny = len(x_grid), len(y_grid)
            U_data = np.zeros((num_steps, Nx, Ny))
            rho_data = np.zeros((num_steps, Nx, Ny))
            sim_times = np.zeros(num_steps)

            for i, ts_key in enumerate(timesteps):
                U_data[i, :, :] = f[f'/TIMESTEPS/{ts_key}/fields/U'][:]
                rho_data[i, :, :] = f[f'/TIMESTEPS/{ts_key}/fields/rho'][:]
                sim_times[i] = f[f'/TIMESTEPS/{ts_key}/stats'].attrs['sim_time_s']

            return {
                "x_grid": x_grid,
                "y_grid": y_grid,
                "U_data": U_data,
                "rho_data": rho_data,
                "sim_times": sim_times,
                "params": params
            }
    except FileNotFoundError:
        print(f"FATAL ERROR: The data file was not found at '{filepath}'.")
        print("Please ensure you have run the 'RUFT_2D_final.py' simulation first.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Plotting Functions ---

def plot_snapshots_2d(data):
    """Creates a static plot of the U field at the start, middle, and end."""
    U = data["U_data"]
    times = data["sim_times"]
    params = data["params"]

    num_steps = U.shape[0]
    indices_to_plot = [0, num_steps // 2, num_steps - 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RUFT 2D Simulation Snapshots: Energy Density (U)", fontsize=16)

    # Determine a common color scale range for all snapshots
    vmin = np.min(U)
    vmax = np.max(U)

    for i, step_idx in enumerate(indices_to_plot):
        ax = axes[i]
        im = ax.imshow(U[step_idx, :, :].T, extent=[0, params['L_x'], 0, params['L_y']],
                       origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Time = {times[step_idx]:.3f} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="U (Unitless)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    output_dir = "VISUALIZATION/notebooks/"
    os.makedirs(output_dir, exist_ok=True)
    snapshot_filename = os.path.join(output_dir, "2d_snapshots.png")
    plt.savefig(snapshot_filename)
    print(f"Snapshot plot saved to '{snapshot_filename}'")

    plt.show()

def create_animation_2d(data):
    """Creates and saves an animation of the Energy Density (U) field."""
    U = data["U_data"]
    times = data["sim_times"]
    params = data["params"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("RUFT 2D: Energy Density (U) Evolution")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Set fixed color scale for stable animation
    vmin = np.min(U)
    vmax = np.max(U)

    # Initial image
    img = ax.imshow(U[0, :, :].T, extent=[0, params['L_x'], 0, params['L_y']],
                    origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(img, label="U (Unitless)")
    time_text = ax.text(0.05, 1.02, '', transform=ax.transAxes, fontsize=12)

    def animate(i):
        img.set_data(U[i, :, :].T)
        time_text.set_text(f'Time = {times[i]:.3f} s')
        return img, time_text

    frame_step = max(1, len(times) // 150) # Aim for ~150 frames
    anim = FuncAnimation(fig, animate, frames=range(0, len(times), frame_step),
                          blit=True, interval=50)

    # Save the animation
    output_dir = "VISUALIZATION/notebooks/"
    os.makedirs(output_dir, exist_ok=True)
    animation_filename = os.path.join(output_dir, "2d_U_field_animation.mp4")
    try:
        anim.save(animation_filename, writer='ffmpeg', fps=15)
        print(f"Animation saved to '{animation_filename}'")
    except Exception as e:
        print("\n--- Animation Saving Failed ---")
        print("Could not save animation. This usually means 'ffmpeg' is not installed or not in your PATH.")
        print(f"System Error: {e}")

    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    simulation_data_2d = load_data_2d(SIM_DATA_PATH)

    if simulation_data_2d:
        print("\n--- Generating 2D Plots ---")
        plot_snapshots_2d(simulation_data_2d)
        create_animation_2d(simulation_data_2d)
        print("\n--- 2D Analysis Complete ---")
