# RUFT-Genesis-Key/analysis/analysis_portal.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import os

# --- Configuration ---
DATA_DIR = "../simulations"
# For portal analysis, we might look at events in 2D or 3D simulations.
# Example: Using fused cores from 2D sim as "portal events"
FUSED_HISTORY_FILE_2D = "2D_fused_history.npy"
ACTIVE_HISTORY_FILE_2D = "2D_active_history.npy"
# Or, if the simulation outputted specific "portal coordinates" or "flux data":
# PORTAL_EVENT_COORDS_FILE = "2D_portal_events_coords.npy" # (time, x, y, strength)
# PORTAL_FLUX_FILE = "3D_flux_data.npy"

# Plotting preferences
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_SIZE_SINGLE = (12, 7)
FIGURE_SIZE_MULTI = (12, 10)
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10

# Analysis parameters
EVENT_PEAK_PROMINENCE = 1  # Prominence for detecting peaks in event timeseries
CLUSTER_EPS = 5.0          # Max distance for DBSCAN clustering of event locations (if spatial data used)
CLUSTER_MIN_SAMPLES = 3    # Min samples for DBSCAN

def load_portal_data(data_dir, filename):
    """Loads data potentially relevant for portal/event analysis."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Warning: Portal data file not found: {path}")
        return None
    try:
        data = np.load(path)
        print(f"Successfully loaded portal-relevant data: {filename}")
        return data
    except Exception as e:
        print(f"Error loading portal data from {filename}: {e}")
        return None

def detect_significant_events(timeseries_data, data_name, prominence=EVENT_PEAK_PROMINENCE, filename_prefix="Portal"):
    """Detects significant events (peaks) in a timeseries."""
    if timeseries_data is None or len(timeseries_data) < 3: # find_peaks needs at least 3 samples
        print(f"Not enough data for event detection in {data_name}.")
        return None, None

    peaks, properties = find_peaks(timeseries_data, prominence=prominence, width=1)

    plt.figure(figsize=FIGURE_SIZE_SINGLE)
    plt.plot(timeseries_data, label=f'{data_name} Timeseries', color='skyblue')
    plt.plot(peaks, timeseries_data[peaks], "x", color='red', markersize=8, label=f'Detected Events ({len(peaks)})')
    plt.title(f'{filename_prefix} - Event Detection in {data_name}', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Time Step', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Magnitude / Count', fontsize=FONT_SIZE_LABEL)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.savefig(f"{filename_prefix}_event_detection_{data_name.lower().replace(' ', '_')}.png")
    print(f"Saved event detection plot for {data_name} to {filename_prefix}_event_detection_{data_name.lower().replace(' ', '_')}.png")
    plt.close()

    if len(peaks) > 0:
        print(f"Detected {len(peaks)} significant events in {data_name} at time steps: {peaks}")
        print(f"Event magnitudes: {timeseries_data[peaks]}")
        print(f"Event prominences: {properties['prominences']}")
        return peaks, properties
    else:
        print(f"No significant events detected in {data_name} with prominence {prominence}.")
        return None, None

def analyze_event_correlations(event_times1, event_times2, name1="Events1", name2="Events2", max_time_diff=10, filename_prefix="Portal"):
    """Analyzes temporal correlations between two sets of event times."""
    if event_times1 is None or event_times2 is None or len(event_times1) == 0 or len(event_times2) == 0:
        print(f"Not enough event data for correlation analysis between {name1} and {name2}.")
        return

    print(f"\n--- Analyzing Event Correlations: {name1} vs {name2} ---")
    correlated_pairs = []
    time_differences = []

    for t1 in event_times1:
        for t2 in event_times2:
            diff = t2 - t1
            if abs(diff) <= max_time_diff:
                correlated_pairs.append((t1, t2))
                time_differences.append(diff)

    if not correlated_pairs:
        print(f"No correlated events found within +/-{max_time_diff} time steps.")
        return

    time_differences = np.array(time_differences)

    plt.figure(figsize=FIGURE_SIZE_SINGLE)
    plt.hist(time_differences, bins=np.arange(-max_time_diff - 0.5, max_time_diff + 1.5, 1), rwidth=0.8, color='mediumpurple')
    plt.title(f'{filename_prefix} - Time Differences Between Correlated {name1} and {name2}', fontsize=FONT_SIZE_TITLE)
    plt.xlabel(f'Time Difference ({name2} time - {name1} time)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Number of Correlated Event Pairs', fontsize=FONT_SIZE_LABEL)
    plt.axvline(0, color='black', linestyle='--', alpha=0.7, label='Simultaneous')
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{filename_prefix}_event_correlation_hist_{name1.lower()}_{name2.lower()}.png")
    print(f"Saved event correlation histogram to {filename_prefix}_event_correlation_hist_{name1.lower()}_{name2.lower()}.png")
    plt.close()

    mean_diff = np.mean(time_differences)
    std_diff = np.std(time_differences)
    print(f"Found {len(correlated_pairs)} correlated event pairs within +/-{max_time_diff} steps.")
    print(f"Mean time difference: {mean_diff:.2f} steps (positive means {name2} occurs after {name1})")
    print(f"Std dev of time difference: {std_diff:.2f} steps")


def analyze_spatial_event_clustering(event_coords_file, filename_prefix="Portal"):
    """
    Analyzes spatial clustering of events if coordinates are available.
    Assumes event_coords_file contains data like (time, x, y, [z], [strength]).
    This is a placeholder for if such data were generated by a simulation.
    """
    event_coords_data = load_portal_data(DATA_DIR, event_coords_file)
    if event_coords_data is None or event_coords_data.ndim != 2 or event_coords_data.shape[1] < 3:
        print(f"Skipping spatial event clustering: Data not found or format incorrect in {event_coords_file}.")
        print("Expected N x (time, x, y, ...) array.")
        return

    print(f"\n--- Analyzing Spatial Event Clustering from {event_coords_file} ---")
    # Assuming columns are [time, x, y, ...] or [time, x, y, z, ...]
    spatial_dims = event_coords_data[:, 1:3] # Default to 2D (x,y)
    if event_coords_data.shape[1] >= 4 and np.all(np.isfinite(event_coords_data[:,3])) : # Check for Z coord
        # Heuristic: if 4th dim looks like a coordinate (not time or large strength values)
        is_z_coord = np.std(event_coords_data[:,3]) < 2 * np.max([np.std(event_coords_data[:,1]),np.std(event_coords_data[:,2])]) if event_coords_data.shape[0]>1 else True
        if is_z_coord and event_coords_data.shape[1] > 3: # if it is a z-coord
             spatial_dims = event_coords_data[:, 1:4] # Use (x,y,z)
        elif event_coords_data.shape[1] == 3 : # if only x,y,z (no time)
             spatial_dims = event_coords_data[:, 0:3]


    if spatial_dims.shape[0] < CLUSTER_MIN_SAMPLES:
        print(f"Not enough event locations ({spatial_dims.shape[0]}) for DBSCAN clustering (min_samples={CLUSTER_MIN_SAMPLES}).")
        return

    print(f"Performing DBSCAN clustering on {spatial_dims.shape[0]} event locations (dimensions: {spatial_dims.shape[1]})...")
    db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(spatial_dims)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f'Estimated number of spatial event clusters: {n_clusters_}')
    print(f'Number of noise points (unclustered events): {n_noise_}')

    # Visualization (2D or 3D scatter plot)
    if spatial_dims.shape[1] == 2: # 2D plot
        plt.figure(figsize=FIGURE_SIZE_SINGLE)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1: col = [0, 0, 0, 1] # Noise is black
            class_member_mask = (labels == k)
            xy = spatial_dims[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=10 if k != -1 else 5, label=f'Cluster {k}' if k != -1 else 'Noise')
        plt.title(f'{filename_prefix} - Spatial Clustering of Events (DBSCAN)', fontsize=FONT_SIZE_TITLE)
        plt.xlabel('X Coordinate', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Y Coordinate', fontsize=FONT_SIZE_LABEL)
        plt.legend(fontsize=FONT_SIZE_LEGEND)
        plt.savefig(f"{filename_prefix}_spatial_event_clusters_2D.png")
        plt.close()
    elif spatial_dims.shape[1] == 3: # 3D plot
        fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)
        ax = fig.add_subplot(111, projection='3d')
        # Similar plotting logic for 3D... (omitted for brevity here but would be analogous)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1: col = [0, 0, 0, 1] # Noise is black
            class_member_mask = (labels == k)
            xyz = spatial_dims[class_member_mask]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o', s=50 if k != -1 else 20, c=[col], label=f'Cluster {k}' if k!=-1 else 'Noise')

        ax.set_title(f'{filename_prefix} - Spatial Clustering of Events (3D DBSCAN)', fontsize=FONT_SIZE_TITLE)
        ax.set_xlabel('X', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Y', fontsize=FONT_SIZE_LABEL)
        ax.set_zlabel('Z', fontsize=FONT_SIZE_LABEL)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        plt.savefig(f"{filename_prefix}_spatial_event_clusters_3D.png")
        plt.close()
    print(f"Saved spatial event cluster plot (if applicable).")


def main_portal_analysis():
    print("--- Starting Portal/Event Analysis ---")

    # Example 1: Using fused cores from 2D simulation as "portal opening events"
    fused_2d_hist = load_portal_data(DATA_DIR, FUSED_HISTORY_FILE_2D)
    if fused_2d_hist is not None:
        print("\nAnalyzing Fused Cores from 2D Simulation as Portal Events:")
        fused_event_times, _ = detect_significant_events(fused_2d_hist, "2D Fused Cores", prominence=1, filename_prefix="Portal_2DFused")

        # If we had another event series, e.g., peaks in active sites
        active_2d_hist = load_portal_data(DATA_DIR, ACTIVE_HISTORY_FILE_2D)
        if active_2d_hist is not None and fused_event_times is not None:
            active_event_times, _ = detect_significant_events(active_2d_hist, "2D Active Bursts", prominence=np.std(active_2d_hist)/2, filename_prefix="Portal_2DActive")
            if active_event_times is not None:
                 analyze_event_correlations(fused_event_times, active_event_times,
                                           name1="FusedCoreEvents", name2="ActiveBurstEvents",
                                           max_time_diff=15, filename_prefix="Portal_2D")
    else:
        print("Skipping analysis of 2D Fused Cores as portal events (data not found).")

    # Example 2: Placeholder for analysis of dedicated portal data (if generated)
    # This would require a simulation that outputs, e.g., coordinates of "portal" formations.
    # PORTAL_EVENT_COORDS_FILE = "2D_portal_events_coords.npy" (example: time, x, y, strength)
    # analyze_spatial_event_clustering(PORTAL_EVENT_COORDS_FILE, filename_prefix="Portal_Spatial")

    # Example 3: Analyzing flux data if available (e.g., from 3D sim)
    # flux_3d_data = load_portal_data(DATA_DIR, "3D_flux_timeseries.npy") # Fictional file
    # if flux_3d_data is not None:
    #     print("\nAnalyzing 3D Flux Data as Portal Activity:")
    #     detect_significant_events(flux_3d_data, "3D Flux Activity", prominence=5, filename_prefix="Portal_3DFlux")

    print("\n--- Portal/Event Analysis Complete ---")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_script_dir, DATA_DIR)

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory '{data_directory}' not found.")
    else:
        main_portal_analysis()
        print("\nTo view plots, check the current directory for .png files.")
        print("Example: Open 'Portal_2DFused_event_detection_2d_fused_cores.png'")
