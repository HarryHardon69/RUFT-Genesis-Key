# RUFT-Genesis-Key/simulations/3D_replicator_final.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

# --- Configuration Parameters ---
GRID_SIZE_X = 30
GRID_SIZE_Y = 30
GRID_SIZE_Z = 30
TIME_STEPS = 100

# Initial state
INITIAL_PATTERN = "central_cube"  # "random", "central_sphere", "central_cube", "multiple_seeds"
INITIAL_DENSITY = 0.05  # For "random"
SEED_COUNT = 5          # For "multiple_seeds"

# Replication and Resource Dynamics
REPLICATION_THRESHOLD_MIN = 4 # Min active neighbors to replicate (out of 26)
REPLICATION_THRESHOLD_MAX = 7 # Max active neighbors (above this, overcrowding)
RESOURCE_INITIAL_LEVEL = 100.0  # Initial resource units per cell
RESOURCE_CONSUMPTION_ACTIVE = 5.0  # Resource consumed by an active cell per step
RESOURCE_CONSUMPTION_REPLICATE = 20.0 # Additional resource consumed for replication
RESOURCE_REGENERATION_RATE = 0.5 # Resource regenerated per cell per step (if not depleted)
DEPLETION_THRESHOLD = 1.0 # Below this, cell cannot be active or replicate

# Cell States
# 0: Inactive / Quiescent (Sufficient resources)
# 1: Active (Consuming resources)
# 2: Depleted (Insufficient resources, cannot activate)
# Refractory period is implicitly handled by resource depletion and regeneration.

# --- Visualization Parameters ---
VISUALIZATION_TYPE = "scatter" # "scatter", "voxels" (voxels can be slow for large grids)
VISUALIZATION_FREQUENCY = 5 # Update plot every N steps (if not using animation)
ANIMATION_ENABLED = True    # If False, plots snapshots
ANIMATION_INTERVAL = 100    # Milliseconds between frames for animation
PLOT_STYLE_ACTIVE = 'cyan'
PLOT_STYLE_DEPLETED = 'red'
PLOT_ALPHA = 0.7

class CellAutomaton3D:
    def __init__(self):
        self.size_x, self.size_y, self.size_z = GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z
        self.grid_state = np.zeros((self.size_x, self.size_y, self.size_z), dtype=int)
        self.grid_resource = np.full_like(self.grid_state, RESOURCE_INITIAL_LEVEL, dtype=float)
        self._initialize_grid()

        self.history_active_cells = []
        self.history_depleted_cells = []
        self.history_total_resource = []

    def _initialize_grid(self):
        if INITIAL_PATTERN == "random":
            self.grid_state = (np.random.rand(self.size_x, self.size_y, self.size_z) < INITIAL_DENSITY).astype(int)
        elif INITIAL_PATTERN == "central_sphere":
            cx, cy, cz = self.size_x // 2, self.size_y // 2, self.size_z // 2
            radius = min(self.size_x, self.size_y, self.size_z) // 4
            for x in range(self.size_x):
                for y in range(self.size_y):
                    for z in range(self.size_z):
                        if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < radius**2:
                            self.grid_state[x, y, z] = 1
        elif INITIAL_PATTERN == "central_cube":
            cx, cy, cz = self.size_x // 2, self.size_y // 2, self.size_z // 2
            s = max(1,min(self.size_x, self.size_y, self.size_z) // 5)
            self.grid_state[cx-s:cx+s, cy-s:cy+s, cz-s:cz+s] = 1
        elif INITIAL_PATTERN == "multiple_seeds":
            for _ in range(SEED_COUNT):
                sx, sy, sz = np.random.randint(0,self.size_x), np.random.randint(0,self.size_y), np.random.randint(0,self.size_z)
                self.grid_state[sx,sy,sz] = 1

        # Cells starting as active immediately consume replication resources if possible, otherwise just active consumption
        for x, y, z in np.argwhere(self.grid_state == 1):
            if self.grid_resource[x,y,z] >= RESOURCE_CONSUMPTION_REPLICATE:
                 self.grid_resource[x,y,z] -= RESOURCE_CONSUMPTION_REPLICATE
            else:
                 self.grid_resource[x,y,z] -= RESOURCE_CONSUMPTION_ACTIVE
            if self.grid_resource[x,y,z] < DEPLETION_THRESHOLD:
                self.grid_state[x,y,z] = 2 # Becomes depleted


    def _count_active_neighbors(self, x, y, z):
        count = 0
        for i in range(max(0, x-1), min(self.size_x, x+2)):
            for j in range(max(0, y-1), min(self.size_y, y+2)):
                for k in range(max(0, z-1), min(self.size_z, z+2)):
                    if (i, j, k) == (x, y, z):
                        continue
                    if self.grid_state[i, j, k] == 1: # Only count active cells
                        count += 1
        return count

    def step(self):
        new_grid_state = self.grid_state.copy()
        new_grid_resource = self.grid_resource.copy()

        for x in range(self.size_x):
            for y in range(self.size_y):
                for z in range(self.size_z):
                    current_state = self.grid_state[x, y, z]
                    current_resource = self.grid_resource[x, y, z]
                    active_neighbors = self._count_active_neighbors(x, y, z)

                    # 1. Resource Regeneration
                    if current_state != 2 : # Depleted cells do not regenerate until neighbors make them active again (implicitly)
                         new_grid_resource[x,y,z] = min(RESOURCE_INITIAL_LEVEL, current_resource + RESOURCE_REGENERATION_RATE)

                    # 2. State Transitions & Resource Consumption
                    if current_state == 1: # Active cell
                        if new_grid_resource[x,y,z] >= RESOURCE_CONSUMPTION_ACTIVE:
                            new_grid_resource[x,y,z] -= RESOURCE_CONSUMPTION_ACTIVE
                            if new_grid_resource[x,y,z] < DEPLETION_THRESHOLD:
                                new_grid_state[x,y,z] = 2 # Becomes depleted
                            # Active cells don't automatically become inactive unless depleted
                            # They might be "killed" by overcrowding or stay active
                        else:
                            new_grid_state[x,y,z] = 2 # Not enough resource to sustain, becomes depleted

                    elif current_state == 0: # Quiescent cell
                        if new_grid_resource[x,y,z] >= RESOURCE_CONSUMPTION_REPLICATE:
                            if REPLICATION_THRESHOLD_MIN <= active_neighbors <= REPLICATION_THRESHOLD_MAX:
                                new_grid_state[x,y,z] = 1 # Replicates (becomes active)
                                new_grid_resource[x,y,z] -= RESOURCE_CONSUMPTION_REPLICATE
                                if new_grid_resource[x,y,z] < DEPLETION_THRESHOLD: # Check if replication depleted it
                                    new_grid_state[x,y,z] = 2
                        elif new_grid_resource[x,y,z] < DEPLETION_THRESHOLD : # handles case where regeneration wasn't enough
                             new_grid_state[x,y,z] = 2 # Not enough to do anything, ensure it's marked depleted.


                    elif current_state == 2: # Depleted cell
                        # Can it become quiescent again due to resource regeneration?
                        if new_grid_resource[x,y,z] >= DEPLETION_THRESHOLD:
                            # Now, can it become active due to neighbors?
                            if new_grid_resource[x,y,z] >= RESOURCE_CONSUMPTION_REPLICATE and \
                               REPLICATION_THRESHOLD_MIN <= active_neighbors <= REPLICATION_THRESHOLD_MAX :
                                new_grid_state[x,y,z] = 1 # Becomes active from depleted
                                new_grid_resource[x,y,z] -= RESOURCE_CONSUMPTION_REPLICATE
                                if new_grid_resource[x,y,z] < DEPLETION_THRESHOLD: # Check again
                                    new_grid_state[x,y,z] = 2
                            elif new_grid_resource[x,y,z] >= DEPLETION_THRESHOLD : # Not enough to replicate or conditions not met
                                 new_grid_state[x,y,z] = 0 # Becomes quiescent
                        # else remains depleted.

        self.grid_state = new_grid_state
        self.grid_resource = new_grid_resource

        self.history_active_cells.append(np.sum(self.grid_state == 1))
        self.history_depleted_cells.append(np.sum(self.grid_state == 2))
        self.history_total_resource.append(np.sum(self.grid_resource))


    def plot_snapshot(self, ax_3d, ax_hist, step_num):
        ax_3d.clear()
        active_cells = np.argwhere(self.grid_state == 1)
        depleted_cells = np.argwhere(self.grid_state == 2)

        if VISUALIZATION_TYPE == "scatter":
            if active_cells.size > 0:
                ax_3d.scatter(active_cells[:,0], active_cells[:,1], active_cells[:,2], color=PLOT_STYLE_ACTIVE, alpha=PLOT_ALPHA, label='Active', s=20)
            if depleted_cells.size > 0:
                ax_3d.scatter(depleted_cells[:,0], depleted_cells[:,1], depleted_cells[:,2], color=PLOT_STYLE_DEPLETED, alpha=PLOT_ALPHA, label='Depleted', s=20)
        elif VISUALIZATION_TYPE == "voxels":
            voxels_display = np.zeros_like(self.grid_state, dtype=bool)
            if active_cells.size > 0:
                voxels_display[active_cells[:,0], active_cells[:,1], active_cells[:,2]] = True
            if depleted_cells.size > 0: # Voxel color is tricky, this will just show presence
                voxels_display[depleted_cells[:,0], depleted_cells[:,1], depleted_cells[:,2]] = True

            # For voxels, we may need to be more creative with colors if showing multiple states
            # This example uses a single color for any non-zero state if 'voxels' is chosen.
            # A better voxel plot would assign colors based on self.grid_state.
            colors = np.empty(self.grid_state.shape, dtype=object)
            colors[self.grid_state == 1] = PLOT_STYLE_ACTIVE
            colors[self.grid_state == 2] = PLOT_STYLE_DEPLETED

            # Only plot if there's something to show
            if np.any(voxels_display):
                 ax_3d.voxels(voxels_display, facecolors=colors[voxels_display], edgecolor='k', alpha=PLOT_ALPHA)


        ax_3d.set_xlim(0, self.size_x)
        ax_3d.set_ylim(0, self.size_y)
        ax_3d.set_zlim(0, self.size_z)
        ax_3d.set_title(f"3D Replicator - Step {step_num}")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        if not ax_3d.get_legend(): # Avoid duplicate legends
            ax_3d.legend(loc='upper left')


        # Update history plots
        ax_hist[0].clear()
        ax_hist[0].plot(self.history_active_cells, 'b-', label='Active Cells')
        ax_hist[0].plot(self.history_depleted_cells, 'r-', label='Depleted Cells')
        ax_hist[0].set_title("Cell Counts Over Time")
        ax_hist[0].set_xlabel("Time Step")
        ax_hist[0].set_ylabel("Count")
        ax_hist[0].legend()
        ax_hist[0].grid(True)

        ax_hist[1].clear()
        ax_hist[1].plot(self.history_total_resource, 'g-', label='Total Resource')
        ax_hist[1].set_title("Total Resource Over Time")
        ax_hist[1].set_xlabel("Time Step")
        ax_hist[1].set_ylabel("Resource Units")
        ax_hist[1].legend()
        ax_hist[1].grid(True)

        plt.draw()
        plt.pause(0.01)

    def run_simulation_no_animation(self):
        print(f"Starting 3D Replicator simulation (no animation): Grid={self.size_x}x{self.size_y}x{self.size_z}, Steps={TIME_STEPS}")
        fig = plt.figure(figsize=(16, 8))
        ax_3d = fig.add_subplot(121, projection='3d')

        # Create a 2x1 grid for history plots on the right
        gs = fig.add_gridspec(2, 2)
        ax_hist_cells = fig.add_subplot(gs[0, 1])
        ax_hist_resource = fig.add_subplot(gs[1, 1])
        ax_hist = [ax_hist_cells, ax_hist_resource]

        fig.tight_layout(pad=3.0)
        start_time = time.time()

        for t in range(TIME_STEPS):
            self.step()
            if (t + 1) % VISUALIZATION_FREQUENCY == 0 or t == 0:
                self.plot_snapshot(ax_3d, ax_hist, t + 1)
                print(f"Step {t + 1}/{TIME_STEPS} - Active: {self.history_active_cells[-1]}, Depleted: {self.history_depleted_cells[-1]}, Resource: {self.history_total_resource[-1]:.2f}")

        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
        final_plot_filename = "3D_replicator_simulation_final_state.png"
        fig.savefig(final_plot_filename)
        print(f"Final state plot saved as {final_plot_filename}")
        plt.show()

    def run_simulation_with_animation(self):
        print(f"Starting 3D Replicator simulation (with animation): Grid={self.size_x}x{self.size_y}x{self.size_z}, Steps={TIME_STEPS}")
        fig = plt.figure(figsize=(16, 8))
        ax_3d = fig.add_subplot(121, projection='3d')

        gs = fig.add_gridspec(2, 2)
        ax_hist_cells = fig.add_subplot(gs[0, 1])
        ax_hist_resource = fig.add_subplot(gs[1, 1])

        fig.tight_layout(pad=3.0)

        # Store artists that change for blitting
        active_scatter = ax_3d.scatter([], [], [], color=PLOT_STYLE_ACTIVE, alpha=PLOT_ALPHA, label='Active', s=20)
        depleted_scatter = ax_3d.scatter([], [], [], color=PLOT_STYLE_DEPLETED, alpha=PLOT_ALPHA, label='Depleted', s=20)

        # Initial legend for 3D plot
        ax_3d.legend(loc='upper left')

        # Lines for history plots
        line_active_hist, = ax_hist_cells.plot([], [], 'b-', label='Active Cells')
        line_depleted_hist, = ax_hist_cells.plot([], [], 'r-', label='Depleted Cells')
        ax_hist_cells.set_title("Cell Counts Over Time")
        ax_hist_cells.set_xlabel("Time Step")
        ax_hist_cells.set_ylabel("Count")
        ax_hist_cells.legend()
        ax_hist_cells.grid(True)
        ax_hist_cells.set_xlim(0, TIME_STEPS)
        ax_hist_cells.set_ylim(0, self.size_x*self.size_y*self.size_z / 4) # Estimate max count

        line_resource_hist, = ax_hist_resource.plot([], [], 'g-', label='Total Resource')
        ax_hist_resource.set_title("Total Resource Over Time")
        ax_hist_resource.set_xlabel("Time Step")
        ax_hist_resource.set_ylabel("Resource Units")
        ax_hist_resource.legend()
        ax_hist_resource.grid(True)
        ax_hist_resource.set_xlim(0, TIME_STEPS)
        ax_hist_resource.set_ylim(0, self.size_x*self.size_y*self.size_z*RESOURCE_INITIAL_LEVEL * 1.1)


        simulation_step_counter = 0

        def update_fig(frame_num):
            nonlocal simulation_step_counter, active_scatter, depleted_scatter
            if simulation_step_counter < TIME_STEPS:
                self.step()

                active_cells = np.argwhere(self.grid_state == 1)
                depleted_cells = np.argwhere(self.grid_state == 2)

                # Update 3D scatter plots
                # Note: For scatter, we need to update the _offsets3d property.
                # This is a bit of a hack for blitting with scatter3d.
                if active_cells.size > 0:
                    active_scatter._offsets3d = (active_cells[:,0], active_cells[:,1], active_cells[:,2])
                else:
                    active_scatter._offsets3d = ([], [], [])

                if depleted_cells.size > 0:
                    depleted_scatter._offsets3d = (depleted_cells[:,0], depleted_cells[:,1], depleted_cells[:,2])
                else:
                    depleted_scatter._offsets3d = ([], [], [])

                ax_3d.set_title(f"3D Replicator - Step {simulation_step_counter + 1}")

                # Update history plots
                line_active_hist.set_data(range(len(self.history_active_cells)), self.history_active_cells)
                line_depleted_hist.set_data(range(len(self.history_depleted_cells)), self.history_depleted_cells)
                if self.history_active_cells or self.history_depleted_cells:
                    max_y_count = max(max(self.history_active_cells if self.history_active_cells else [0]),
                                      max(self.history_depleted_cells if self.history_depleted_cells else [0]))
                    ax_hist_cells.set_ylim(0, max(10, max_y_count * 1.1))


                line_resource_hist.set_data(range(len(self.history_total_resource)), self.history_total_resource)
                if self.history_total_resource:
                     ax_hist_resource.set_ylim(0, max(10,max(self.history_total_resource) *1.1))


                if (simulation_step_counter + 1) % VISUALIZATION_FREQUENCY == 0:
                    print(f"Step {simulation_step_counter + 1}/{TIME_STEPS} - Active: {self.history_active_cells[-1]}, Depleted: {self.history_depleted_cells[-1]}, Resource: {self.history_total_resource[-1]:.2f}")

                simulation_step_counter += 1

            # Return list of artists that have been modified
            return [active_scatter, depleted_scatter, line_active_hist, line_depleted_hist, line_resource_hist, ax_3d.title]


        start_time = time.time()
        ani = animation.FuncAnimation(fig, update_fig, frames=TIME_STEPS, interval=ANIMATION_INTERVAL, blit=True, repeat=False)

        plt.show()
        end_time = time.time()
        print(f"Animation displayed. Total time: {end_time - start_time:.2f} seconds.")

        try:
            animation_filename = "3D_replicator_simulation.gif" # or .mp4
            ani.save(animation_filename, writer='imagemagick', fps=1000/ANIMATION_INTERVAL)
            print(f"Animation saved as {animation_filename}")
        except Exception as e:
            print(f"Could not save animation: {e}. Make sure imagemagick (for gif) or ffmpeg (for mp4) is installed.")

        # Save final data
        np.save("3D_active_history.npy", np.array(self.history_active_cells))
        np.save("3D_depleted_history.npy", np.array(self.history_depleted_cells))
        np.save("3D_resource_history.npy", np.array(self.history_total_resource))
        print("Active, depleted, and resource history data saved.")


if __name__ == "__main__":
    automaton = CellAutomaton3D()
    if ANIMATION_ENABLED:
        automaton.run_simulation_with_animation()
    else:
        automaton.run_simulation_no_animation()
    print("3D Replicator simulation run complete.")

    # To load data:
    # active_hist = np.load("3D_active_history.npy")
    # depleted_hist = np.load("3D_depleted_history.npy")
    # resource_hist = np.load("3D_resource_history.npy")
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(active_hist, label="Active")
    # ax[0].plot(depleted_hist, label="Depleted")
    # ax[0].legend()
    # ax[1].plot(resource_hist, label="Total Resource", color='g')
    # ax[1].legend()
    # plt.show()
