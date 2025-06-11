# RUFT-Genesis-Key/analysis/analysis_2d.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from scipy.stats import linregress, entropy as scipy_entropy
import os

# --- Configuration ---
DATA_DIR = "../simulations"  # Relative path to where simulation data is stored
ACTIVE_HISTORY_FILE_2D = "2D_active_history.npy"
FUSED_HISTORY_FILE_2D = "2D_fused_history.npy"
ENTROPY_HISTORY_FILE_2D = "2D_entropy_history.npy" # Assuming this also exists for 2D

# Plotting preferences (consistent with 1D analysis script)
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_SIZE_SINGLE = (10, 6)
FIGURE_SIZE_MULTI = (12, 10) # For plots with multiple subplots
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10

# Analysis parameters
PEAK_PROMINENCE = 5
CORR_LAG_MAX = 100
STABILITY_WINDOW_SIZE = 30 # Window for calculating rolling std dev for stability

def load_2d_simulation_data(data_dir):
    """Loads 2D simulation data."""
    active_path = os.path.join(data_dir, ACTIVE_HISTORY_FILE_2D)
    fused_path = os.path.join(data_dir, FUSED_HISTORY_FILE_2D)
    entropy_path = os.path.join(data_dir, ENTROPY_HISTORY_FILE_2D)

    data_loaded = {}
    paths = {
        "active": active_path,
        "fused": fused_path,
        "entropy": entropy_path
    }
    all_found = True
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"Warning: {key.capitalize()} history file not found: {path}")
            data_loaded[key] = None # Still add key but with None value
            if key != "entropy": # Entropy is optional for some analyses here
                 all_found = False # fused and active are more critical for core 2D plots
        else:
            try:
                data_loaded[key] = np.load(path)
                print(f"Successfully loaded 2D {key} data from {path}")
            except Exception as e:
                print(f"Error loading 2D {key} data from {path}: {e}")
                data_loaded[key] = None
                all_found = False

    if not all_found and (data_loaded["active"] is None or data_loaded["fused"] is None) :
        print("Critical 2D data (active or fused sites) missing. Some analyses may fail or be skipped.")

    return data_loaded.get("active"), data_loaded.get("fused"), data_loaded.get("entropy")


def plot_2d_timeseries(active_history, fused_history, entropy_history, filename_prefix="2D"):
    """Plots active sites, fused cores, and entropy over time for 2D data."""
    num_plots = 2 + (1 if entropy_history is not None else 0)
    fig, ax = plt.subplots(num_plots, 1, figsize=(FIGURE_SIZE_MULTI[0], num_plots * 4), sharex=True)

    plot_idx = 0

    if active_history is not None:
        ax[plot_idx].plot(active_history, label='Active Sites', color='deepskyblue')
        ax[plot_idx].set_title(f'{filename_prefix} - Active Sites Over Time', fontsize=FONT_SIZE_TITLE)
        ax[plot_idx].set_ylabel('Number of Active Sites', fontsize=FONT_SIZE_LABEL)
        ax[plot_idx].legend(fontsize=FONT_SIZE_LEGEND)
        plot_idx += 1
    else:
        print("Skipping active sites plot: data not available.")
        if num_plots == 1 : # only entropy was available
            ax.set_title('Active Sites Data Missing', fontsize=FONT_SIZE_TITLE) # use the single ax

    if fused_history is not None:
        ax[plot_idx].plot(fused_history, label='Fused Cores', color='orangered')
        ax[plot_idx].set_title(f'{filename_prefix} - Fused Cores Over Time', fontsize=FONT_SIZE_TITLE)
        ax[plot_idx].set_ylabel('Number of Fused Cores', fontsize=FONT_SIZE_LABEL)
        ax[plot_idx].legend(fontsize=FONT_SIZE_LEGEND)
        plot_idx += 1
    else:
        print("Skipping fused cores plot: data not available.")
        if num_plots == 1 and plot_idx == 0:
             ax.set_title('Fused Cores Data Missing', fontsize=FONT_SIZE_TITLE)
        elif num_plots > 1 and plot_idx < len(ax):
             ax[plot_idx].set_title('Fused Cores Data Missing', fontsize=FONT_SIZE_TITLE)


    if entropy_history is not None:
        ax[plot_idx].plot(entropy_history, label='Shannon Entropy', color='mediumseagreen')
        ax[plot_idx].set_title(f'{filename_prefix} - Shannon Entropy Over Time', fontsize=FONT_SIZE_TITLE)
        ax[plot_idx].set_ylabel('Entropy (bits)', fontsize=FONT_SIZE_LABEL)
        ax[plot_idx].legend(fontsize=FONT_SIZE_LEGEND)
        # Max entropy for 3 states (0: inactive, 1: active, 2: fused) is log2(3)
        ax[plot_idx].set_ylim(0, np.log2(3) + 0.1 if np.max(entropy_history) < np.log2(3) + 0.05 else np.max(entropy_history) * 1.1)
        plot_idx +=1
    else:
        print("Skipping entropy plot: data not available.")
        if num_plots == 1 and plot_idx == 0:
             ax.set_title('Entropy Data Missing', fontsize=FONT_SIZE_TITLE)
        elif num_plots > 1 and plot_idx < len(ax):
             ax[plot_idx].set_title('Entropy Data Missing', fontsize=FONT_SIZE_TITLE)


    # Set common X label only on the last plot
    if plot_idx > 0:
        ax[plot_idx-1].set_xlabel('Time Step', fontsize=FONT_SIZE_LABEL)
    elif isinstance(ax, np.ndarray): # handles case where ax is array but no plots made
        ax[0].set_xlabel('Time Step', fontsize=FONT_SIZE_LABEL)
    else: # handles case where ax is single axis and no plots made
        ax.set_xlabel('Time Step', fontsize=FONT_SIZE_LABEL)


    plt.tight_layout(pad=2.0)
    plt.savefig(f"{filename_prefix}_basic_timeseries.png")
    print(f"Saved 2D basic timeseries plot to {filename_prefix}_basic_timeseries.png")
    plt.close(fig)

# Re-use periodicity and trend analysis functions from 1D analysis if they are generic enough
# For now, let's copy and adapt slightly for clarity if needed.
def analyze_2d_periodicity(data, data_name, filename_prefix="2D"):
    """Analyzes periodicity using autocorrelation for 2D data components."""
    if data is None or len(data) < 2 * CORR_LAG_MAX :
        print(f"Not enough data for 2D {data_name} periodicity analysis.")
        return

    detrended_data = data - np.mean(data)
    autocorr = correlate(detrended_data, detrended_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if np.var(detrended_data) > 1e-6:
        autocorr /= (np.var(detrended_data) * len(detrended_data))
    else:
        autocorr = np.ones_like(autocorr)

    lags = np.arange(len(autocorr))

    plt.figure(figsize=FIGURE_SIZE_SINGLE)
    plt.plot(lags[:CORR_LAG_MAX], autocorr[:CORR_LAG_MAX], label=f'Autocorrelation of {data_name}', color='darkcyan')
    plt.title(f'{filename_prefix} - Autocorrelation for {data_name}', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Lag (Time Steps)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Normalized Autocorrelation', fontsize=FONT_SIZE_LABEL)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(f"{filename_prefix}_autocorrelation_{data_name.lower().replace(' ', '_')}.png")
    print(f"Saved 2D autocorrelation plot for {data_name} to {filename_prefix}_autocorrelation_{data_name.lower().replace(' ', '_')}.png")
    plt.close()

    peaks, _ = find_peaks(autocorr[:CORR_LAG_MAX], prominence=0.1, distance=5)
    if len(peaks) > 0:
        print(f"Potential periods for 2D {data_name}: {peaks}")
    else:
        print(f"No significant periodic behavior detected for 2D {data_name} via autocorrelation.")


def analyze_2d_trends(histories, history_names, filename_prefix="2D"):
    """Analyzes trends using linear regression for multiple 2D data components."""
    if not histories or not history_names or len(histories) != len(history_names):
        print("Invalid input for 2D trend analysis.")
        return

    fig, ax = plt.subplots(len(histories), 1, figsize=(FIGURE_SIZE_MULTI[0], len(histories)*4), sharex=True)
    if len(histories) == 1: # if only one history, ax is not an array
        ax = [ax]

    for i, history in enumerate(histories):
        name = history_names[i]
        if history is None or len(history) < 2:
            print(f"Skipping trend analysis for {name}: Not enough data.")
            ax[i].text(0.5, 0.5, f'No data for {name}', horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)
            continue

        time_steps = np.arange(len(history))
        slope, intercept, r_value, p_value, _ = linregress(time_steps, history)

        print(f"\nTrend analysis for {name} ({filename_prefix}):")
        print(f"  Slope: {slope:.4f}")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  P-value: {p_value:.4f}")

        ax[i].plot(time_steps, history, label=name, color=plt.cm.viridis(i / len(histories)))
        ax[i].plot(time_steps, intercept + slope * time_steps, linestyle='--', color='black', label=f'Trend (R²={r_value**2:.2f})')
        ax[i].set_title(f'{filename_prefix} - Trend in {name}', fontsize=FONT_SIZE_TITLE)
        ax[i].set_ylabel('Count / Value', fontsize=FONT_SIZE_LABEL)
        ax[i].legend(fontsize=FONT_SIZE_LEGEND)

    ax[-1].set_xlabel('Time Step', fontsize=FONT_SIZE_LABEL)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{filename_prefix}_trends_analysis.png")
    print(f"Saved 2D trends analysis plot to {filename_prefix}_trends_analysis.png")
    plt.close(fig)


def plot_2d_phase_space(active_history, fused_history, entropy_history, filename_prefix="2D"):
    """Plots phase space diagrams for 2D data:
       1. Fused Cores vs. Active Sites
       2. Entropy vs. Active Sites (if entropy available)
       3. Entropy vs. Fused Cores (if entropy available)
    """
    if active_history is None or fused_history is None:
        print("Skipping phase space plots: Active or Fused history data is missing.")
        return

    # Plot 1: Fused Cores vs. Active Sites
    plt.figure(figsize=FIGURE_SIZE_SINGLE)
    time_points = np.arange(len(active_history)) # Assuming active and fused have same length
    scatter1 = plt.scatter(active_history, fused_history, c=time_points, cmap='magma', alpha=0.7, s=10)
    plt.title(f'{filename_prefix} - Phase Space: Fused Cores vs. Active Sites', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Number of Active Sites', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Number of Fused Cores', fontsize=FONT_SIZE_LABEL)
    cbar1 = plt.colorbar(scatter1, label='Time Step')
    cbar1.ax.tick_params(labelsize=FONT_SIZE_LABEL-2)
    cbar1.set_label('Time Step', size=FONT_SIZE_LABEL)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{filename_prefix}_phase_space_active_fused.png")
    print(f"Saved phase space plot (Active vs. Fused) to {filename_prefix}_phase_space_active_fused.png")
    plt.close()

    if entropy_history is not None and len(entropy_history) == len(active_history):
        # Plot 2: Entropy vs. Active Sites
        plt.figure(figsize=FIGURE_SIZE_SINGLE)
        scatter2 = plt.scatter(active_history, entropy_history, c=time_points, cmap='viridis', alpha=0.7, s=10)
        plt.title(f'{filename_prefix} - Phase Space: Entropy vs. Active Sites', fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Number of Active Sites', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Shannon Entropy (bits)', fontsize=FONT_SIZE_LABEL)
        cbar2 = plt.colorbar(scatter2, label='Time Step')
        cbar2.ax.tick_params(labelsize=FONT_SIZE_LABEL-2)
        cbar2.set_label('Time Step', size=FONT_SIZE_LABEL)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{filename_prefix}_phase_space_entropy_active.png")
        print(f"Saved phase space plot (Entropy vs. Active) to {filename_prefix}_phase_space_entropy_active.png")
        plt.close()

        # Plot 3: Entropy vs. Fused Cores
        plt.figure(figsize=FIGURE_SIZE_SINGLE)
        scatter3 = plt.scatter(fused_history, entropy_history, c=time_points, cmap='plasma', alpha=0.7, s=10)
        plt.title(f'{filename_prefix} - Phase Space: Entropy vs. Fused Cores', fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Number of Fused Cores', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Shannon Entropy (bits)', fontsize=FONT_SIZE_LABEL)
        cbar3 = plt.colorbar(scatter3, label='Time Step')
        cbar3.ax.tick_params(labelsize=FONT_SIZE_LABEL-2)
        cbar3.set_label('Time Step', size=FONT_SIZE_LABEL)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{filename_prefix}_phase_space_entropy_fused.png")
        print(f"Saved phase space plot (Entropy vs. Fused) to {filename_prefix}_phase_space_entropy_fused.png")
        plt.close()
    else:
        print("Skipping entropy-related phase space plots: entropy data not available or mismatched length.")

def main_2d_analysis():
    print("--- Starting 2D Simulation Analysis ---")

    active_hist, fused_hist, entropy_hist = load_2d_simulation_data(DATA_DIR)

    # Basic plots
    plot_2d_timeseries(active_hist, fused_hist, entropy_hist, filename_prefix="2D")

    # Periodicity analysis
    if active_hist is not None:
        analyze_2d_periodicity(active_hist, "Active Sites", filename_prefix="2D")
    if fused_hist is not None:
        analyze_2d_periodicity(fused_hist, "Fused Cores", filename_prefix="2D")
    if entropy_hist is not None:
        analyze_2d_periodicity(entropy_hist, "Entropy", filename_prefix="2D")

    # Trend analysis
    histories_for_trend = []
    history_names_for_trend = []
    if active_hist is not None:
        histories_for_trend.append(active_hist)
        history_names_for_trend.append("Active Sites")
    if fused_hist is not None:
        histories_for_trend.append(fused_hist)
        history_names_for_trend.append("Fused Cores")
    if entropy_hist is not None:
        histories_for_trend.append(entropy_hist)
        history_names_for_trend.append("Entropy")

    if histories_for_trend:
         analyze_2d_trends(histories_for_trend, history_names_for_trend, filename_prefix="2D")
    else:
        print("No data available for trend analysis.")


    # Phase space plots
    plot_2d_phase_space(active_hist, fused_hist, entropy_hist, filename_prefix="2D")

    # Example of stability analysis (can reuse/adapt from 1D)
    # stability_active_2d = calculate_stability(active_hist, window_size=STABILITY_WINDOW_SIZE, data_name="2D Active Sites")
    # if stability_active_2d is not None:
    #     # Plotting for stability_active_2d...
    #     pass

    print("\n--- 2D Simulation Analysis Complete ---")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_script_dir, DATA_DIR)

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory '{data_directory}' not found.")
    else:
        main_2d_analysis()
        print("\nTo view plots, check the current directory for .png files.")
        print("Example: Open '2D_basic_timeseries.png'")
