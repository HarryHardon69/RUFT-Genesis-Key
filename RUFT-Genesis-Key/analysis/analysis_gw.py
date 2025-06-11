import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
from scipy.signal import welch
from astropy import constants as const
from astropy import units as u

# --- Configuration & Constants ---
G_SI = const.G
C_SI = const.c
R_OBSERVER_SI = 1 * u.kpc

# --- Data Loading & Processing (Polished) ---
def load_sim_data(filepath):
    """Loads all necessary data from a RUFT simulation HDF5 file."""
    print(f"--- Loading data from {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            p = dict(f['/SIMULATION_PARAMETERS'].attrs)
            x = f['/MESH_DATA/x_grid'][:]
            y = f['/MESH_DATA/y_grid'][:]
            time_history = f['/time_history'][:]

            timesteps_group = f['/TIMESTEPS']
            ts_keys = sorted(timesteps_group.keys(), key=int)

            if not ts_keys:
                print("ERROR: No timestep data found in the HDF5 file.")
                return None, None, None, None, None

            print(f"Found {len(ts_keys)} saved timesteps.")

            U_list = []
            data_is_complete = True
            for ts_key in ts_keys:
                # Check if the required 'U' dataset exists
                if f'{ts_key}/fields/U' in timesteps_group:
                    U_list.append(timesteps_group[f'{ts_key}/fields/U'][:])
                else:
                    data_is_complete = False
                    break # Exit loop if data is missing

            if not data_is_complete:
                # --- THIS IS THE CONDITIONAL WARNING ---
                print("\n*** WARNING: Full U-field time series not found in this HDF5 file. ***")
                print("This analysis will proceed by RECONSTRUCTING a mock U-field.")
                print("This is NOT a physically rigorous result and only tests the analysis pipeline.\n")

                initial_rho = 0.1 * p['rho_0'] * np.exp(-((np.meshgrid(x, y, indexing='ij')[0] - p['Lx']/2)**2 +
                                                       (np.meshgrid(x, y, indexing='ij')[1] - p['Ly']/2)**2) / (p['Lx']/8)**2)
                tau_E_hist = f['/tau_E_history'][:]
                amplitude_modulation = tau_E_hist / np.max(tau_E_hist) if np.max(tau_E_hist) > 0 else np.zeros_like(tau_E_hist)
                U_series = np.array([initial_rho * amp * np.cos(p['omega'] * t) for t, amp in zip(time_history, amplitude_modulation)])
            else:
                print("Full U-field time series loaded successfully.")
                U_series = np.array(U_list)

            return U_series, x, y, time_history, p

    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        return None, None, None, None, None

# ... (The rest of the functions: calculate_quadrupole_moment, calculate_gw_strain, plot_gw_results are identical)
def calculate_quadrupole_moment(U_series, x, y, p):
    print("Calculating time-varying quadrupole moment...")
    dx, dy = p['dx'], p['dy']
    q_ij = np.zeros((U_series.shape[0], 2, 2))
    X, Y = np.meshgrid(x, y, indexing='ij')
    ENERGY_DENSITY_FACTOR = 1e-5
    I_xx = np.sum(U_series * ENERGY_DENSITY_FACTOR * X**2, axis=(1, 2)) * dx * dy
    I_yy = np.sum(U_series * ENERGY_DENSITY_FACTOR * Y**2, axis=(1, 2)) * dx * dy
    I_xy = np.sum(U_series * ENERGY_DENSITY_FACTOR * X*Y, axis=(1, 2)) * dx * dy
    q_ij[:, 0, 0] = I_xx; q_ij[:, 1, 1] = I_yy; q_ij[:, 0, 1] = q_ij[:, 1, 0] = I_xy
    return q_ij

def calculate_gw_strain(q_ij, times):
    print("Calculating gravitational wave strain h(t)...")
    if len(times) < 3: return None, None
    dt = np.mean(np.diff(times));
    if dt == 0: return None, None
    q_dot = np.gradient(q_ij, dt, axis=0); q_ddot = np.gradient(q_dot, dt, axis=0)
    prefactor = (G_SI.value / (C_SI.value**4 * R_OBSERVER_SI.to(u.m).value))
    h_plus = prefactor * (q_ddot[:, 0, 0] - q_ddot[:, 1, 1])
    h_cross = prefactor * (2 * q_ddot[:, 0, 1])
    return h_plus, h_cross

def plot_gw_results(times, h_plus, h_cross, run_id):
    print("Generating GW plots...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"RUFT Predicted Gravitational Wave Signature\nRun ID: {run_id}", fontsize=16)
    axes[0].set_title(f"Waveform as seen at {R_OBSERVER_SI.to(u.kpc):.1f}")
    axes[0].plot(times, h_plus, label='h_plus polarization'); axes[0].plot(times, h_cross, label='h_cross polarization', alpha=0.7)
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Strain (h)"); axes[0].grid(True, linestyle='--'); axes[0].legend()
    fs = 1.0 / np.mean(np.diff(times))
    freqs, psd_plus = welch(h_plus, fs=fs, nperseg=len(h_plus)//4 if len(h_plus) > 8 else len(h_plus))
    char_strain = np.sqrt(freqs * psd_plus)
    axes[1].loglog(freqs, char_strain)
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Characteristic Strain (h_c)")
    axes[1].set_title("Frequency Spectrum"); axes[1].grid(True, which="both", ls="--")
    axes[1].set_xlim(left=1, right=fs/2 if fs > 2 else 2); axes[1].set_ylim(bottom=1e-25)
    output_dir = "VISUALIZATION/notebooks/"
    plot_filename = os.path.join(output_dir, f"gw_analysis_{run_id}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(plot_filename)
    print(f"GW analysis plot saved to '{plot_filename}'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze RUFT simulation data for Gravitational Waves.")
    parser.add_argument("filepath", type=str, help="Path to the HDF5 simulation output file.")
    args = parser.parse_args()
    run_id = os.path.basename(args.filepath).replace("output_", "").replace(".h5", "")
    U_series, x, y, times, params = load_sim_data(args.filepath)
    if U_series is not None and len(U_series) > 0:
        q_ij = calculate_quadrupole_moment(U_series, x, y, params)
        h_plus, h_cross = calculate_gw_strain(q_ij, times)
        if h_plus is not None:
            plot_gw_results(times, h_plus, h_cross, run_id)
            print("\n--- GW Analysis Complete ---")
    else:
        print("Could not proceed with analysis due to data loading issues.")

if __name__ == "__main__":
    main()
