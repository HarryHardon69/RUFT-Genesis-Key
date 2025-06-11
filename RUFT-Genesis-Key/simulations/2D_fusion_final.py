# RUFT-Genesis-Key/simulations/2D_fusion_final.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
import time

# --- Configuration Parameters ---
GRID_WIDTH = 100  # Width of the 2D grid
GRID_HEIGHT = 100 # Height of the 2D grid
TIME_STEPS = 300   # Number of simulation steps

# Initial state patterns
INITIAL_PATTERN_TYPE = "random_clusters" # "random", "central_block", "dual_sources", "random_clusters"
INITIAL_DENSITY = 0.15  # For "random" and "random_clusters"
CLUSTER_COUNT = 5       # For "random_clusters"
CLUSTER_RADIUS = 8      # For "random_clusters"

# Activation and Fusion Parameters
ACTIVATION_THRESHOLD_MIN = 0.12  # Minimum normalized sum of active neighbors to activate
ACTIVATION_THRESHOLD_MAX = 0.30  # Maximum normalized sum (above this, overcrowding inhibits activation)
FUSION_THRESHOLD = 0.60          # Normalized sum of active neighbors for a site to become a "fused" core
REFRACTORY_PERIOD = 10            # Steps a site remains inactive (state 0) after activation or fusion
FUSED_CORE_STRENGTH = 2.0        # Contribution of a fused core to its neighbors' activation sum (amplification)
FUSED_CORE_DURATION = 25         # Steps a fused core remains in its special state (state 2)

# Neighborhood kernel (Moore neighborhood)
NEIGHBORHOOD_KERNEL = np.array([[1, 1, 1],
                                [1, 0, 1], # Central 0 means don't count self in sum
                                [1, 1, 1]])
NORMALIZATION_FACTOR = np.sum(NEIGHBORHOOD_KERNEL) # For normalizing neighbor sum

BOUNDARY_CONDITIONS = "periodic" # "periodic", "fixed"

# --- Visualization Parameters ---
VISUALIZATION_FREQUENCY = 5  # Update plot every N steps (if not using animation)
ANIMATION_ENABLED = True
ANIMATION_INTERVAL = 50  # Milliseconds between frames
PLOT_STYLE = 'plasma'    # Colormap: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray'

# --- State Definitions ---
# 0: Inactive (resting or refractory)
# 1: Active
# 2: Fused Core

class CellAutomaton2D:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.refractory_timers = np.zeros_like(self.grid, dtype=int)
        self.fused_core_timers = np.zeros_like(self.grid, dtype=int)

        self.active_history = []
        self.fused_history = []
        self.entropy_history = []

        self._initialize_grid()

    def _initialize_grid(self):
        if INITIAL_PATTERN_TYPE == "random":
            self.grid = (np.random.rand(self.height, self.width) < INITIAL_DENSITY).astype(int)
        elif INITIAL_PATTERN_TYPE == "central_block":
            cx, cy = self.width // 2, self.height // 2
            size = 5
            self.grid[cy-size//2:cy+size//2+1, cx-size//2:cx+size//2+1] = 1
        elif INITIAL_PATTERN_TYPE == "dual_sources":
            self.grid[self.height // 4, self.width // 4] = 1
            self.grid[3 * self.height // 4, 3 * self.width // 4] = 1
            self.grid[self.height // 2, 3 * self.width // 4] = 1
        elif INITIAL_PATTERN_TYPE == "random_clusters":
            for _ in range(CLUSTER_COUNT):
                cx, cy = np.random.randint(0, self.width), np.random.randint(0, self.height)
                for r in range(self.height):
                    for c in range(self.width):
                        if (r-cy)**2 + (c-cx)**2 < CLUSTER_RADIUS**2:
                             if np.random.rand() < INITIAL_DENSITY * 5: # Higher density within cluster
                                self.grid[r,c] = 1
        else: # Default to random
            self.grid = (np.random.rand(self.height, self.width) < INITIAL_DENSITY).astype(int)

        # Initially active cells start their refractory period countdown
        self.refractory_timers[self.grid == 1] = REFRACTORY_PERIOD


    def _get_neighbor_influence(self):
        # Create a map of influences: active cells (1) contribute 1, fused cores (2) contribute FUSED_CORE_STRENGTH
        influence_map = np.zeros_like(self.grid, dtype=float)
        influence_map[self.grid == 1] = 1.0
        influence_map[self.grid == 2] = FUSED_CORE_STRENGTH

        # Convolve to get sum of influences
        mode = 'wrap' if BOUNDARY_CONDITIONS == "periodic" else 'same'
        neighbor_influence_sum = convolve2d(influence_map, NEIGHBORHOOD_KERNEL, mode=mode, boundary='fill', fillvalue=0)

        if NORMALIZATION_FACTOR > 0:
            normalized_influence = neighbor_influence_sum / NORMALIZATION_FACTOR
        else:
            normalized_influence = np.zeros_like(neighbor_influence_sum)
        return normalized_influence

    def step(self):
        new_grid = self.grid.copy()

        # Decrement timers
        self.refractory_timers[self.refractory_timers > 0] -= 1
        self.fused_core_timers[self.fused_core_timers > 0] -= 1

        # Reset fused cores whose duration has expired
        expired_fused_cores = (self.grid == 2) & (self.fused_core_timers == 0)
        new_grid[expired_fused_cores] = 0 # Become inactive
        self.refractory_timers[expired_fused_cores] = REFRACTORY_PERIOD # Enter refractory

        neighbor_influence = self._get_neighbor_influence()

        for r in range(self.height):
            for c in range(self.width):
                current_state = self.grid[r, c]
                influence = neighbor_influence[r, c]

                # --- State transitions ---
                if current_state == 1: # Currently Active
                    new_grid[r, c] = 0 # Becomes inactive (will enter refractory)
                    self.refractory_timers[r, c] = REFRACTORY_PERIOD
                    # Check for fusion potential based on this step's influence
                    if influence >= FUSION_THRESHOLD:
                        new_grid[r, c] = 2 # Transition to Fused Core
                        self.fused_core_timers[r, c] = FUSED_CORE_DURATION
                        self.refractory_timers[r, c] = 0 # Fused cores are not refractory immediately

                elif current_state == 0: # Currently Inactive
                    if self.refractory_timers[r, c] > 0:
                        continue # Still in refractory, remains inactive

                    # Check for activation
                    if ACTIVATION_THRESHOLD_MIN <= influence <= ACTIVATION_THRESHOLD_MAX:
                        new_grid[r, c] = 1 # Activate
                        self.refractory_timers[r, c] = REFRACTORY_PERIOD # Set refractory for next step

                elif current_state == 2: # Currently Fused Core
                    # Stays fused core until its timer runs out (handled by expired_fused_cores logic)
                    pass


        self.grid = new_grid
        self.active_history.append(np.sum(self.grid == 1))
        self.fused_history.append(np.sum(self.grid == 2))
        self.entropy_history.append(self.calculate_shannon_entropy())

    def calculate_shannon_entropy(self):
        p_states = [np.mean(self.grid == i) for i in range(3)] # Probabilities for state 0, 1, 2
        entropy = 0.0
        for p in p_states:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def run_simulation_no_animation(self):
        print(f"Starting 2D Fusion simulation (no animation): Grid={self.width}x{self.height}, Steps={TIME_STEPS}")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        im = ax1.imshow(self.grid, cmap=PLOT_STYLE, vmin=0, vmax=2)
        ax1.set_title(f"2D Fusion CA - Step 0")
        plt.colorbar(im, ax=ax1, ticks=[0, 1, 2], label="State (0:Inactive, 1:Active, 2:Fused)")

        line_active, = ax2.plot([], [], 'b-', label='Active Sites')
        ax2.set_title("Active Sites")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Count")
        ax2.set_xlim(0, TIME_STEPS)

        line_fused, = ax3.plot([], [], 'r-', label='Fused Cores')
        ax3.set_title("Fused Cores")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Count")
        ax3.set_xlim(0, TIME_STEPS)

        line_entropy, = ax4.plot([],[], 'g-', label='Entropy')
        ax4.set_title("Shannon Entropy")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Entropy (bits)")
        ax4.set_xlim(0, TIME_STEPS)
        ax4.set_ylim(0, np.log2(3) + 0.1) # Max entropy for 3 states

        fig.tight_layout(pad=2.0)
        start_time = time.time()

        for t in range(TIME_STEPS):
            self.step()
            if (t + 1) % VISUALIZATION_FREQUENCY == 0 or t == 0:
                im.set_data(self.grid)
                ax1.set_title(f"2D Fusion CA - Step {t + 1}")

                line_active.set_data(range(len(self.active_history)), self.active_history)
                ax2.set_ylim(0, max(10, np.max(self.active_history) * 1.1 if self.active_history else 10) )

                line_fused.set_data(range(len(self.fused_history)), self.fused_history)
                ax3.set_ylim(0, max(10, np.max(self.fused_history) * 1.1 if self.fused_history else 10) )

                line_entropy.set_data(range(len(self.entropy_history)), self.entropy_history)

                plt.draw()
                plt.pause(0.01)
                print(f"Step {t + 1}/{TIME_STEPS} - Active: {self.active_history[-1]}, Fused: {self.fused_history[-1]}, Entropy: {self.entropy_history[-1]:.4f}")

        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
        final_plot_filename = "2D_fusion_simulation_final_state.png"
        fig.savefig(final_plot_filename)
        print(f"Final state plot saved as {final_plot_filename}")
        plt.show()


    def run_simulation_with_animation(self):
        print(f"Starting 2D Fusion simulation (with animation): Grid={self.width}x{self.height}, Steps={TIME_STEPS}")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        im = ax1.imshow(self.grid, cmap=PLOT_STYLE, vmin=0, vmax=2, animated=True)
        ax1.set_title(f"2D Fusion CA - Step 0")
        plt.colorbar(im, ax=ax1, ticks=[0, 1, 2], label="State (0:Inactive, 1:Active, 2:Fused)")

        line_active, = ax2.plot([], [], 'b-', label='Active Sites')
        ax2.set_title("Active Sites")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Count")
        ax2.set_xlim(0, TIME_STEPS)

        line_fused, = ax3.plot([], [], 'r-', label='Fused Cores')
        ax3.set_title("Fused Cores")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Count")
        ax3.set_xlim(0, TIME_STEPS)

        line_entropy, = ax4.plot([],[], 'g-', label='Entropy')
        ax4.set_title("Shannon Entropy")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Entropy (bits)")
        ax4.set_xlim(0, TIME_STEPS)
        ax4.set_ylim(0, np.log2(3) + 0.1) # Max entropy for 3 states

        fig.tight_layout(pad=2.0)

        simulation_step_counter = 0 # Need a mutable counter for the animation function

        def update_fig(frame_num):
            nonlocal simulation_step_counter # Use nonlocal to modify the outer scope variable
            # For animation, self.step() is called 'frame_num' times before this.
            # We want to call it once per frame.
            if simulation_step_counter < TIME_STEPS : # only step if simulation is not over
                self.step()
                im.set_array(self.grid)
                ax1.set_title(f"2D Fusion CA - Step {simulation_step_counter + 1}")

                line_active.set_data(range(len(self.active_history)), self.active_history)
                ax2.set_ylim(0, max(10, np.max(self.active_history) * 1.1 if self.active_history else 10) )


                line_fused.set_data(range(len(self.fused_history)), self.fused_history)
                ax3.set_ylim(0, max(10, np.max(self.fused_history) * 1.1 if self.fused_history else 10) )

                line_entropy.set_data(range(len(self.entropy_history)), self.entropy_history)

                if (simulation_step_counter + 1) % VISUALIZATION_FREQUENCY == 0:
                     print(f"Step {simulation_step_counter + 1}/{TIME_STEPS} - Active: {self.active_history[-1]}, Fused: {self.fused_history[-1]}, Entropy: {self.entropy_history[-1]:.4f}")
                simulation_step_counter +=1


            return im, line_active, line_fused, line_entropy

        start_time = time.time()
        # Number of frames for animation will be TIME_STEPS
        ani = animation.FuncAnimation(fig, update_fig, frames=TIME_STEPS, interval=ANIMATION_INTERVAL, blit=True, repeat=False)

        plt.show() # This will run the animation

        end_time = time.time()
        print(f"Simulation (animation displayed) finished in {end_time - start_time:.2f} seconds.")

        # Save the animation (optional, can be slow)
        try:
            animation_filename = "2D_fusion_simulation.gif" # or .mp4
            ani.save(animation_filename, writer='imagemagick', fps=1000/ANIMATION_INTERVAL) # or 'ffmpeg' for mp4
            print(f"Animation saved as {animation_filename}")
        except Exception as e:
            print(f"Could not save animation: {e}. Make sure imagemagick (for gif) or ffmpeg (for mp4) is installed.")

        # Save final data
        np.save("2D_active_history.npy", np.array(self.active_history))
        np.save("2D_fused_history.npy", np.array(self.fused_history))
        np.save("2D_entropy_history.npy", np.array(self.entropy_history))
        print("Active, fused, and entropy history data saved.")


if __name__ == "__main__":
    automaton = CellAutomaton2D()
    if ANIMATION_ENABLED:
        automaton.run_simulation_with_animation()
    else:
        automaton.run_simulation_no_animation()
    print("2D Fusion simulation run complete.")

    # To load data:
    # active_history = np.load("2D_active_history.npy")
    # fused_history = np.load("2D_fused_history.npy")
    # entropy_history = np.load("2D_entropy_history.npy")
    # plt.plot(active_history, label='Active')
    # plt.plot(fused_history, label='Fused')
    # plt.legend()
    # plt.figure()
    # plt.plot(entropy_history, label='Entropy')
    # plt.legend()
    # plt.show()
