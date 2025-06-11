import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- Configuration ---
# This script is designed to be run from the root of the "RUFT v4.0" project directory.
# It will automatically find the output file from the 1D solver.
SIM_DATA_PATH = os.path.join("SIMULATIONS", "1D_Solver", "output_1d.h5")

# --- Data Loading Function ---
def load_data(filepath):
    """Loads and returns data from the specified HDF5 file."""
    print(f"--- Loading data from {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            # Load mesh data
            x_grid = f['/MESH_DATA/x_grid'][:]

            # Load parameters for context
            params = dict(f['/SIMULATION_PARAMETERS'].attrs)

            # Load timesteps
            timesteps = sorted(f['/TIMESTEPS'].keys(), key=int)
            print(f"Found {len(timesteps)} saved timesteps.")

            # Pre-allocate arrays for performance
            num_steps = len(timesteps)
            Nx = len(x_grid)
            U_data = np.zeros((num_steps, Nx))
            rho_data = np.zeros((num_steps, Nx))
            sim_times = np.zeros(num_steps)

            for i, ts_key in enumerate(timesteps):
                U_data[i, :] = f[f'/TIMESTEPS/{ts_key}/fields/U'][:]
                rho_data[i, :] = f[f'/TIMESTEPS/{ts_key}/fields/rho'][:]
                sim_times[i] = f[f'/TIMESTEPS/{ts_key}/stats'].attrs['sim_time_s']

            return {
                "x_grid": x_grid,
                "U_data": U_data,
                "rho_data": rho_data,
                "sim_times": sim_times,
                "params": params
            }
    except FileNotFoundError:
        print(f"FATAL ERROR: The data file was not found at '{filepath}'.")
        print("Please ensure you have run the 'RUFT_1D_final.py' simulation first.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Plotting Functions ---

def plot_snapshots(data):
    """Creates a static plot of the fields at the start, middle, and end of the simulation."""
    x = data["x_grid"]
    U = data["U_data"]
    rho = data["rho_data"]
    times = data["sim_times"]

    num_steps = U.shape[0]
    indices_to_plot = [0, num_steps // 2, num_steps - 1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("RUFT 1D Simulation Snapshots", fontsize=16)

    # Plot U (Energy Density)
    axes[0].set_title("Energy Density (U)")
    for i in indices_to_plot:
        axes[0].plot(x, U[i, :], label=f'Time = {times[i]:.3f} s')
    axes[0].set_ylabel("U (Unitless)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Plot rho (Mass Density)
    axes[1].set_title("Mass Density (rho)")
    for i in indices_to_plot:
        axes[1].plot(x, rho[i, :], label=f'Time = {times[i]:.3f} s')
    axes[1].set_xlabel("Position (m)")
    axes[1].set_ylabel("rho (Unitless)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    output_dir = "VISUALIZATION/notebooks/"
    os.makedirs(output_dir, exist_ok=True)
    snapshot_filename = os.path.join(output_dir, "1d_snapshots.png")
    plt.savefig(snapshot_filename)
    print(f"Snapshot plot saved to '{snapshot_filename}'")

    plt.show()

def create_animation(data):
    """Creates and saves an animation of the Energy Density (U) field."""
    x = data["x_grid"]
    U = data["U_data"]
    times = data["sim_times"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("RUFT 1D: Energy Density (U) Evolution")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("U (Unitless)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set fixed y-limits for a stable animation view
    y_min = np.min(U) * 1.1
    y_max = np.max(U) * 1.1
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(np.min(x), np.max(x))

    line, = ax.plot(x, U[0, :], lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        line.set_ydata(U[i, :])
        time_text.set_text(f'Time = {times[i]:.3f} s')
        return line, time_text

    # Create the animation
    # Note: We use a subset of frames for a smoother video if there are many saved steps.
    frame_step = max(1, len(times) // 200) # Aim for ~200 frames in the video
    anim = FuncAnimation(fig, animate, frames=range(0, len(times), frame_step),
                          blit=True, interval=50)

    # Save the animation
    output_dir = "VISUALIZATION/notebooks/"
    os.makedirs(output_dir, exist_ok=True)
    animation_filename = os.path.join(output_dir, "1d_U_field_animation.mp4")
    try:
        anim.save(animation_filename, writer='ffmpeg', fps=15)
        print(f"Animation saved to '{animation_filename}'")
    except Exception as e:
        print("\n--- Animation Saving Failed ---")
        print("Could not save animation. This usually means 'ffmpeg' is not installed.")
        print("To install on Debian/Ubuntu: sudo apt-get install ffmpeg")
        print("To install on other systems, see ffmpeg documentation.")
        print(f"System Error: {e}")

    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    simulation_data = load_data(SIM_DATA_PATH)

    if simulation_data:
        print("\n--- Generating Plots ---")
        plot_snapshots(simulation_data)
        create_animation(simulation_data)
        print("\n--- Analysis Complete ---")
