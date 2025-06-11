# RUFT-Genesis-Key/analysis/analysis_1d.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from scipy.stats import linregress, entropy as scipy_entropy
import os

# --- Configuration ---
DATA_DIR = "../simulations"  # Relative path to where simulation data is stored
ACTIVE_HISTORY_FILE_1D = "1D_active_history.npy"
ENTROPY_HISTORY_FILE_1D = "1D_entropy_history.npy"

# Plotting preferences
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_SIZE_SINGLE = (10, 6)
FIGURE_SIZE_MULTI = (12, 10)
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10

# Analysis parameters
PEAK_PROMINENCE = 5  # For peak detection in active sites
CORR_LAG_MAX = 100   # Max lag for autocorrelation

def load_simulation_data(data_dir, active_file, entropy_file):
    """Loads 1D simulation data."""
    active_path = os.path.join(data_dir, active_file)
    entropy_path = os.path.join(data_dir, entropy_file)

    if not os.path.exists(active_path):
        print(f"Error: Active history file not found: {active_path}")
        return None, None
    if not os.path.exists(entropy_path):
        print(f"Error: Entropy history file not found: {entropy_path}")
        return None, None

    try:
        active_history = np.load(active_path)
        entropy_history = np.load(entropy_path)
        print(f"Successfully loaded data: {active_file}, {entropy_file}")
        return active_history, entropy_history
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def plot_basic_timeseries(active_history, entropy_history, filename_prefix="1D"):
    """Plots active sites and entropy over time."""
    if active_history is None or entropy_history is None:
        return

    fig, ax = plt.subplots(2, 1, figsize=FIGURE_SIZE_MULTI, sharex=True)

    # Active sites
    ax[0].plot(active_history, label='Active Sites', color='royalblue')
    ax[0].set_title(f'{filename_prefix} - Active Sites Over Time', fontsize=FONT_SIZE_TITLE)
    ax[0].set_ylabel('Number of Active Sites', fontsize=FONT_SIZE_LABEL)
    ax[0].legend(fontsize=FONT_SIZE_LEGEND)

    # Entropy
    ax[1].plot(entropy_history, label='Shannon Entropy', color='forestgreen')
    ax[1].set_title(f'{filename_prefix} - Shannon Entropy Over Time', fontsize=FONT_SIZE_TITLE)
    ax[1].set_xlabel('Time Step', fontsize=FONT_SIZE_LABEL)
    ax[1].set_ylabel('Entropy (bits)', fontsize=FONT_SIZE_LABEL)
    ax[1].legend(fontsize=FONT_SIZE_LEGEND)
    ax[1].set_ylim(0, 1.05) # Max entropy for binary system is 1

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_basic_timeseries.png")
    print(f"Saved basic timeseries plot to {filename_prefix}_basic_timeseries.png")
    plt.close(fig)

def analyze_periodicity(data, data_name, filename_prefix="1D"):
    """Analyzes periodicity using autocorrelation."""
    if data is None or len(data) < 2 * CORR_LAG_MAX:
        print(f"Not enough data for {data_name} periodicity analysis.")
        return

    # Detrend data (simple mean subtraction)
    detrended_data = data - np.mean(data)

    autocorr = correlate(detrended_data, detrended_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:] # Keep only positive lags

    # Normalize
    if np.var(detrended_data) > 1e-6 : # Avoid division by zero for constant data
        autocorr /= (np.var(detrended_data) * len(detrended_data))
    else: # if data is constant, autocorrelation is flat
        autocorr = np.ones_like(autocorr)


    lags = np.arange(len(autocorr))

    plt.figure(figsize=FIGURE_SIZE_SINGLE)
    plt.plot(lags[:CORR_LAG_MAX], autocorr[:CORR_LAG_MAX], label=f'Autocorrelation of {data_name}', color='purple')
    plt.title(f'{filename_prefix} - Autocorrelation Analysis for {data_name}', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Lag (Time Steps)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Normalized Autocorrelation', fontsize=FONT_SIZE_LABEL)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(f"{filename_prefix}_autocorrelation_{data_name.lower().replace(' ', '_')}.png")
    print(f"Saved autocorrelation plot for {data_name} to {filename_prefix}_autocorrelation_{data_name.lower().replace(' ', '_')}.png")
    plt.close()

    # Find peaks in autocorrelation (potential periods)
    peaks, _ = find_peaks(autocorr[:CORR_LAG_MAX], prominence=0.1, distance=5)
    if len(peaks) > 0:
        print(f"Potential periods for {data_name} (from autocorrelation peaks): {peaks}")
    else:
        print(f"No significant periodic behavior detected for {data_name} via autocorrelation.")


def analyze_trends(active_history, entropy_history, filename_prefix="1D"):
    """Analyzes trends using linear regression."""
    if active_history is None or entropy_history is None:
        return

    time_steps = np.arange(len(active_history))

    # Trend for active sites
    slope_active, intercept_active, r_value_active, p_value_active, _ = linregress(time_steps, active_history)
    print(f"\nTrend analysis for Active Sites ({filename_prefix}):")
    print(f"  Slope: {slope_active:.4f}")
    print(f"  R-squared: {r_value_active**2:.4f}")
    print(f"  P-value: {p_value_active:.4f}")

    # Trend for entropy
    slope_entropy, intercept_entropy, r_value_entropy, p_value_entropy, _ = linregress(time_steps, entropy_history)
    print(f"\nTrend analysis for Entropy ({filename_prefix}):")
    print(f"  Slope: {slope_entropy:.4f}")
    print(f"  R-squared: {r_value_entropy**2:.4f}")
    print(f"  P-value: {p_value_entropy:.4f}")

    # Plot trends
    fig, ax = plt.subplots(2, 1, figsize=FIGURE_SIZE_MULTI, sharex=True)
    ax[0].plot(time_steps, active_history, label='Active Sites', color='lightblue')
    ax[0].plot(time_steps, intercept_active + slope_active * time_steps, color='darkblue', linestyle='--', label=f'Trend (R²={r_value_active**2:.2f})')
    ax[0].set_title(f'{filename_prefix} - Trend in Active Sites', fontsize=FONT_SIZE_TITLE)
    ax[0].set_ylabel('Number of Active Sites', fontsize=FONT_SIZE_LABEL)
    ax[0].legend(fontsize=FONT_SIZE_LEGEND)

    ax[1].plot(time_steps, entropy_history, label='Shannon Entropy', color='lightgreen')
    ax[1].plot(time_steps, intercept_entropy + slope_entropy * time_steps, color='darkgreen', linestyle='--', label=f'Trend (R²={r_value_entropy**2:.2f})')
    ax[1].set_title(f'{filename_prefix} - Trend in Shannon Entropy', fontsize=FONT_SIZE_TITLE)
    ax[1].set_xlabel('Time Step', fontsize=FONT_SIZE_LABEL)
    ax[1].set_ylabel('Entropy (bits)', fontsize=FONT_SIZE_LABEL)
    ax[1].legend(fontsize=FONT_SIZE_LEGEND)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_trends_analysis.png")
    print(f"Saved trends analysis plot to {filename_prefix}_trends_analysis.png")
    plt.close(fig)


def calculate_stability(data, window_size=50, data_name="data"):
    """Calculates stability as the inverse of rolling standard deviation."""
    if data is None or len(data) < window_size:
        print(f"Not enough data for {data_name} stability analysis.")
        return None

    rolling_std = np.convolve(data, np.ones(window_size)/window_size, mode='valid') # Moving average
    rolling_std = np.array([np.std(data[i:i+window_size]) for i in range(len(data) - window_size + 1)])

    # Stability: inverse of std. Add small epsilon to avoid division by zero.
    stability = 1.0 / (rolling_std + 1e-9)

    print(f"\nStability analysis for {data_name}:")
    print(f"  Mean rolling standard deviation: {np.mean(rolling_std):.4f}")
    print(f"  Mean stability: {np.mean(stability):.4f}")
    return stability


def plot_phase_space(active_history, entropy_history, filename_prefix="1D"):
    """Plots a phase space diagram of entropy vs. active sites."""
    if active_history is None or entropy_history is None:
        return

    plt.figure(figsize=FIGURE_SIZE_SINGLE)
    # Use a colormap to show time evolution
    time_points = np.arange(len(active_history))
    scatter = plt.scatter(active_history, entropy_history, c=time_points, cmap='viridis', alpha=0.7, s=10)

    plt.title(f'{filename_prefix} - Phase Space: Entropy vs. Active Sites', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Number of Active Sites', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Shannon Entropy (bits)', fontsize=FONT_SIZE_LABEL)
    cbar = plt.colorbar(scatter, label='Time Step')
    cbar.ax.tick_params(labelsize=FONT_SIZE_LABEL-2)
    cbar.set_label('Time Step', size=FONT_SIZE_LABEL)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{filename_prefix}_phase_space.png")
    print(f"Saved phase space plot to {filename_prefix}_phase_space.png")
    plt.close()

def main_1d_analysis():
    print("--- Starting 1D Simulation Analysis ---")

    active_hist, entropy_hist = load_simulation_data(DATA_DIR, ACTIVE_HISTORY_FILE_1D, ENTROPY_HISTORY_FILE_1D)

    if active_hist is None or entropy_hist is None:
        print("Exiting due to data loading issues.")
        return

    # Basic plots
    plot_basic_timeseries(active_hist, entropy_hist, filename_prefix="1D")

    # Periodicity analysis
    analyze_periodicity(active_hist, "Active Sites", filename_prefix="1D")
    analyze_periodicity(entropy_hist, "Entropy", filename_prefix="1D")

    # Trend analysis
    analyze_trends(active_hist, entropy_hist, filename_prefix="1D")

    # Stability analysis (using active sites as an example)
    stability_active = calculate_stability(active_hist, data_name="Active Sites")
    if stability_active is not None:
        plt.figure(figsize=FIGURE_SIZE_SINGLE)
        plt.plot(np.arange(len(stability_active)), stability_active, label='Stability of Active Sites', color='teal')
        plt.title('1D - Stability of Active Sites Over Time', fontsize=FONT_SIZE_TITLE)
        plt.xlabel(f'Time Step (windowed, size={50})', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Stability (1 / Rolling Std Dev)', fontsize=FONT_SIZE_LABEL)
        plt.legend(fontsize=FONT_SIZE_LEGEND)
        plt.savefig("1D_stability_active_sites.png")
        print("Saved stability plot for active sites to 1D_stability_active_sites.png")
        plt.close()

    # Phase space plot
    plot_phase_space(active_hist, entropy_hist, filename_prefix="1D")

    print("\n--- 1D Simulation Analysis Complete ---")


if __name__ == "__main__":
    # Ensure the ../simulations directory exists relative to this script
    # This is a common structure, but might need adjustment based on execution context
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_script_dir, DATA_DIR)

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory '{data_directory}' not found. Please ensure the simulation data is present.")
        print("You might need to run the simulations first, or adjust DATA_DIR if they are elsewhere.")
    else:
        main_1d_analysis()
        print("\nTo view plots, check the current directory for .png files.")
        print("Example: Open '1D_basic_timeseries.png'")
