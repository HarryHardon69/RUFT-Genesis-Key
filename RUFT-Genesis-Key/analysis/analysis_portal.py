import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Data Loading Function (Unchanged) ---
def load_portal_data(filepath):
    print(f"--- [Analysis Portal] Loading data from {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            p = dict(f['/SIMULATION_PARAMETERS'].attrs)
            x_grid = f['/MESH_DATA/x_grid'][:]; y_grid = f['/MESH_DATA/y_grid'][:]
            timesteps_group = f['/TIMESTEPS']
            ts_keys = sorted(timesteps_group.keys(), key=int)
            print(f"Found {len(ts_keys)} saved timesteps.")
            U_series = np.array([timesteps_group[f'{key}/fields/U'][:] for key in ts_keys])
            phi_series = np.array([timesteps_group[f'{key}/fields/phi'][:] for key in ts_keys])
            sim_times = np.array([timesteps_group[f'{key}/stats'].attrs['sim_time_s'] for key in ts_keys])
            return {"x_grid": x_grid, "y_grid": y_grid, "U_series": U_series, "phi_series": phi_series, "sim_times": sim_times, "params": p}
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}"); return None

# --- NEW: Snapshot Plotting Function ---
def plot_portal_snapshots(data, run_id):
    """Creates a static plot of the final state of the Phi and U fields."""
    print("Generating final state snapshot plot...")
    phi = data["phi_series"][-1]; U = data["U_series"][-1] # Get the last frame
    p = data["params"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"RUFT Entangled Aperture - Final State\nRun ID: {run_id}", fontsize=16)

    # Plot Phi Field
    vmax_phi = np.max(np.abs(phi)); vmin_phi = -vmax_phi
    im1 = axes[0].imshow(phi.T, origin='lower', cmap='viridis', extent=[0, p['Lx'], 0, p['Ly']], vmin=vmin_phi, vmax=vmax_phi)
    axes[0].set_title("Final State: Phi Field (\u03C6)")
    fig.colorbar(im1, ax=axes[0], label="\u03C6 (Unitless)")

    # Plot U Field
    vmax_U = np.max(np.abs(U)); vmin_U = -vmax_U
    im2 = axes[1].imshow(U.T, origin='lower', cmap='viridis', extent=[0, p['Lx'], 0, p['Ly']], vmin=vmin_U, vmax=vmax_U)
    axes[1].set_title("Final State: U Field (Spacetime Perturbation)")
    fig.colorbar(im2, ax=axes[1], label="U (Unitless)")

    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)
    snapshot_filename = os.path.join(output_dir, f"snapshot_portal_{run_id}.png")
    plt.savefig(snapshot_filename)
    print(f"Snapshot plot saved to '{snapshot_filename}'")
    plt.close(fig) # Close the figure to prevent display issues

# --- REFINED: Animation Function ---
def create_portal_animation(data, run_id):
    """Creates and saves an animation of the Phi and U fields."""
    print("Generating portal animation...")
    phi = data["phi_series"]; U = data["U_series"]
    times = data["sim_times"]; p = data["params"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"RUFT Entangled Aperture Evolution\nRun ID: {run_id}", fontsize=16)

    # --- Setup for both plots ---
    axes[0].set_title("Phi Field (\u03C6)"); axes[1].set_title("U Field (Spacetime Perturbation)")
    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    vmax_phi = np.max(np.abs(phi)); vmin_phi = -vmax_phi
    img_phi = axes[0].imshow(phi[0].T, origin='lower', cmap='viridis', extent=[0, p['Lx'], 0, p['Ly']], vmin=vmin_phi, vmax=vmax_phi)
    fig.colorbar(img_phi, ax=axes[0], label="\u03C6 (Unitless)")

    vmax_U = np.max(np.abs(U)); vmin_U = -vmax_U
    img_U = axes[1].imshow(U[0].T, origin='lower', cmap='viridis', extent=[0, p['Lx'], 0, p['Ly']], vmin=vmin_U, vmax=vmax_U)
    fig.colorbar(img_U, ax=axes[1], label="U (Unitless)")

    time_text = fig.text(0.5, 0.9, '', ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    plt.tight_layout(rect=[0, 0.05, 1, 0.9])

    # Animation update function
    def animate(i):
        img_phi.set_data(phi[i].T); img_U.set_data(U[i].T)
        time_text.set_text(f'Time = {times[i]:.3f} s')
        return img_phi, img_U, time_text

    anim = FuncAnimation(fig, animate, frames=U.shape[0], blit=True, interval=50)

    # Save the animation
    output_dir = "analysis"
    animation_filename = os.path.join(output_dir, f"animation_portal_{run_id}.mp4")
    try:
        anim.save(animation_filename, writer='ffmpeg', fps=15, dpi=150)
        print(f"Animation successfully saved to '{animation_filename}'")
    except Exception as e:
        print(f"\n--- Animation Saving Failed. Could not save to '{animation_filename}'. 'ffmpeg' may not be installed. ---")
        print(f"System Error: {e}")

    # FIX: Close the plot window after saving to prevent the error loop
    plt.close(fig)

def main():
    """Main execution function."""
    print("--- RUFT v5.0 Genesis Key: 2D Portal Data Analyzer ---")
    parser = argparse.ArgumentParser(description="Generate visualizations from a RUFT Portal simulation file.")
    default_path = os.path.join("simulations", "output", "output_portal_v2.1_w150_C1.0.h5")
    parser.add_argument("filepath", type=str, nargs='?', default=default_path, help=f"Path to HDF5 file. Defaults to '{default_path}'.")
    args = parser.parse_args()
    run_id = os.path.basename(args.filepath).replace("output_", "").replace(".h5", "")

    data = load_portal_data(args.filepath)
    if data:
        print("\n--- Generating Visualizations ---")
        # Run both functions separately for robust execution
        plot_portal_snapshots(data, run_id)
        create_portal_animation(data, run_id)
        print("\n--- Portal Analysis Complete ---")

if __name__ == "__main__":
    main()
