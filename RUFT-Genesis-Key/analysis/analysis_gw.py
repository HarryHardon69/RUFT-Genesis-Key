# RUFT-Genesis-Key/analysis/analysis_gw.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import os

# --- Configuration ---
DATA_DIR = "../simulations" # Data source
# Assuming there might be specific data files for "GW" analysis, e.g., from the 3D sim
# For now, let's assume we might use aggregate data like total active cells from 3D
ACTIVE_HISTORY_FILE_3D = "3D_active_history.npy"
# Or, a dedicated file for "field strength" or similar if the simulation produces it.
# GW_DATA_FILE = "3D_gw_potential_field.npy"

# Plotting preferences
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_SIZE_SINGLE = (12, 7)
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10

# Analysis parameters for PSD (Power Spectral Density)
PSD_SEGMENT_LENGTH = 256 # Length of segments for Welch's method
PSD_OVERLAP_FACTOR = 0.5 # Overlap factor for segments

def load_gw_data(data_dir, filename):
    """Loads data potentially relevant for GW-like analysis."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Error: GW data file not found: {path}")
        return None
    try:
        data = np.load(path)
        print(f"Successfully loaded GW-relevant data: {filename}")
        return data
    except Exception as e:
        print(f"Error loading GW data from {filename}: {e}")
        return None

def plot_power_spectral_density(timeseries_data, data_name, sampling_rate=1.0, filename_prefix="GW"):
    """
    Calculates and plots the Power Spectral Density (PSD) of a timeseries.
    Sampling rate is 1.0 assuming data is per time step.
    """
    if timeseries_data is None or len(timeseries_data) < PSD_SEGMENT_LENGTH:
        print(f"Not enough data for PSD analysis of {data_name}. Minimum {PSD_SEGMENT_LENGTH} points needed.")
        return

    nperseg = min(PSD_SEGMENT_LENGTH, len(timeseries_data))
    noverlap = int(nperseg * PSD_OVERLAP_FACTOR)

    frequencies, psd = welch(timeseries_data - np.mean(timeseries_data), # Detrend
                               fs=sampling_rate,
                               nperseg=nperseg,
                               noverlap=noverlap)

    plt.figure(figsize=FIGURE_SIZE_SINGLE)
    plt.semilogy(frequencies, psd, color='crimson')
    plt.title(f'{filename_prefix} - Power Spectral Density of {data_name}', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Frequency (cycles/time step)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('PSD (Power/Frequency Unit)', fontsize=FONT_SIZE_LABEL)
    plt.grid(True, which="both", ls="-", alpha=0.7)

    # Highlight dominant frequencies
    peaks, _ = find_peaks(psd, prominence=np.max(psd)/10) # Prominence relative to max power
    if len(peaks) > 0:
        plt.plot(frequencies[peaks], psd[peaks], "x", color='blue', markersize=8, label=f'Dominant Frequencies ({len(peaks)})')
        print(f"Dominant frequencies for {data_name} (from PSD peaks): {frequencies[peaks]}")
    plt.legend(fontsize=FONT_SIZE_LEGEND)

    plt.savefig(f"{filename_prefix}_psd_{data_name.lower().replace(' ', '_')}.png")
    print(f"Saved PSD plot for {data_name} to {filename_prefix}_psd_{data_name.lower().replace(' ', '_')}.png")
    plt.close()
    return frequencies, psd

def analyze_gw_patterns(data_source_1, data_source_2=None, source_names=["Source1", "Source2"], filename_prefix="GW_Pattern"):
    """
    Placeholder for analyzing patterns that might be metaphorically GW-like.
    This could involve looking for propagating waves, synchronized bursts, etc.
    For now, this might involve cross-correlation if two relevant time series are provided.
    """
    if data_source_1 is None:
        print("Primary data source for GW pattern analysis is missing.")
        return

    print(f"\n--- GW Pattern Analysis ({filename_prefix}) ---")
    print(f"Analyzing {source_names[0]}...")
    # Example: Autocorrelation of the primary source (similar to periodicity)
    # (Code adapted from 1D analysis)
    if len(data_source_1) > 20: # Min length for meaningful autocorrelation
        detrended_data = data_source_1 - np.mean(data_source_1)
        autocorr = correlate(detrended_data, detrended_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if np.var(detrended_data) > 1e-6:
            autocorr /= (np.var(detrended_data) * len(detrended_data))
        else:
            autocorr = np.ones_like(autocorr)

        lags = np.arange(len(autocorr))
        plt.figure(figsize=FIGURE_SIZE_SINGLE)
        plt.plot(lags[:CORR_LAG_MAX], autocorr[:CORR_LAG_MAX], label=f'Autocorrelation of {source_names[0]}', color='indigo')
        plt.title(f'{filename_prefix} - Autocorrelation of {source_names[0]}', fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Lag', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Normalized Autocorrelation', fontsize=FONT_SIZE_LABEL)
        plt.legend(fontsize=FONT_SIZE_LEGEND)
        plt.savefig(f"{filename_prefix}_autocorr_{source_names[0].lower()}.png")
        plt.close()
        print(f"Saved autocorrelation for {source_names[0]}.")


    if data_source_1 is not None and data_source_2 is not None:
        print(f"Analyzing cross-correlation between {source_names[0]} and {source_names[1]}...")
        if len(data_source_1) != len(data_source_2):
            print("Error: Data sources for cross-correlation must have the same length.")
            # Optionally, truncate to the shorter length if appropriate for the analysis
            # min_len = min(len(data_source_1), len(data_source_2))
            # data_source_1 = data_source_1[:min_len]
            # data_source_2 = data_source_2[:min_len]
            return

        detrended_1 = data_source_1 - np.mean(data_source_1)
        detrended_2 = data_source_2 - np.mean(data_source_2)

        # Ensure variance is not zero before normalizing
        std_1 = np.std(detrended_1)
        std_2 = np.std(detrended_2)

        if std_1 < 1e-6 or std_2 < 1e-6:
            print("One or both data sources have near-zero variance. Cross-correlation may not be meaningful.")
            # cross_corr = np.zeros(len(detrended_1) * 2 -1) # Or handle differently
            # For simplicity, we'll just plot the raw correlation if normalization is problematic
            cross_corr = correlate(detrended_1, detrended_2, mode='full')
            norm_factor = 1.0 # No normalization
        else:
            cross_corr = correlate(detrended_1, detrended_2, mode='full')
            # Normalize cross-correlation
            norm_factor = std_1 * std_2 * len(detrended_1)
            cross_corr = cross_corr / norm_factor


        lags = np.arange(-len(detrended_1) + 1, len(detrended_1))

        plt.figure(figsize=FIGURE_SIZE_SINGLE)
        plt.plot(lags, cross_corr, label=f'Cross-correlation: {source_names[0]} vs {source_names[1]}', color='darkorange')
        plt.title(f'{filename_prefix} - Cross-correlation Analysis', fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Lag (Time Steps)', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Normalized Cross-correlation' if norm_factor != 1.0 else 'Cross-correlation', fontsize=FONT_SIZE_LABEL)
        plt.axvline(0, color='black', linestyle='--', alpha=0.5) # Zero lag line
        plt.legend(fontsize=FONT_SIZE_LEGEND)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.savefig(f"{filename_prefix}_cross_correlation_{source_names[0].lower()}_{source_names[1].lower()}.png")
        print(f"Saved cross-correlation plot to {filename_prefix}_cross_correlation_{source_names[0].lower()}_{source_names[1].lower()}.png")
        plt.close()

        max_corr_idx = np.argmax(np.abs(cross_corr))
        print(f"Max absolute cross-correlation is {cross_corr[max_corr_idx]:.4f} at lag {lags[max_corr_idx]}.")

    else:
        print("Second data source not provided; skipping cross-correlation.")


def main_gw_analysis():
    print("--- Starting GW-like Feature Analysis ---")

    # Example: Use total active cells from 3D simulation as a proxy for "system energy"
    # This is a placeholder; actual GW analysis would need specific data types
    # from simulations designed to model such phenomena.
    active_3d_hist = load_gw_data(DATA_DIR, ACTIVE_HISTORY_FILE_3D)
    # entropy_3d_hist = load_gw_data(DATA_DIR, "3D_entropy_history.npy") # If available and relevant

    if active_3d_hist is None:
        print("Primary data (3D active history) for GW analysis not available. Limited analysis possible.")
    else:
        # Plot the primary timeseries
        plt.figure(figsize=FIGURE_SIZE_SINGLE)
        plt.plot(active_3d_hist, label='Total Active Cells (3D Sim)', color='teal')
        plt.title('GW Analysis - Input: Total Active Cells from 3D Simulation', fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Time Step', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Number of Active Cells', fontsize=FONT_SIZE_LABEL)
        plt.legend(fontsize=FONT_SIZE_LEGEND)
        plt.savefig("GW_input_timeseries_3D_active.png")
        plt.close()

        # Analyze Power Spectral Density
        plot_power_spectral_density(active_3d_hist, "Total Active Cells (3D)", filename_prefix="GW")

        # Placeholder for more advanced GW-like pattern analysis
        # This could involve looking for propagating waves if spatial data is loaded,
        # or analyzing event timings.
        # For now, let's use the active_3d_hist for autocorrelation as an example.
        # If another relevant timeseries was available, e.g., "3D_total_resource.npy",
        # it could be used for cross-correlation.
        resource_3d_hist = load_gw_data(DATA_DIR, "3D_resource_history.npy") # Example second source

        source_names_for_pattern = ["3D Active Cells"]
        data_sources_for_pattern = [active_3d_hist]
        if resource_3d_hist is not None:
            source_names_for_pattern.append("3D Total Resource")
            # Ensure same length for cross-correlation if used
            min_len = min(len(active_3d_hist), len(resource_3d_hist))
            data_sources_for_pattern = [active_3d_hist[:min_len], resource_3d_hist[:min_len]]
            analyze_gw_patterns(data_sources_for_pattern[0], data_sources_for_pattern[1], source_names_for_pattern, filename_prefix="GW_Pattern_ActiveResource")
        else:
            analyze_gw_patterns(data_sources_for_pattern[0], source_names=source_names_for_pattern, filename_prefix="GW_Pattern_ActiveOnly")


    print("\n--- GW-like Feature Analysis Complete ---")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_script_dir, DATA_DIR)

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory '{data_directory}' not found.")
    else:
        main_gw_analysis()
        print("\nTo view plots, check the current directory for .png files.")
        print("Example: Open 'GW_psd_total_active_cells_3d.png'")
