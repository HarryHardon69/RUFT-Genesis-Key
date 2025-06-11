# RUFT-Genesis-Key/simulations/1D_final.py

import numpy as np
import matplotlib.pyplot as plt
import time

# --- Configuration Parameters ---
GRID_SIZE = 200  # Size of the 1D grid
TIME_STEPS = 500  # Number of simulation steps
INITIAL_DENSITY = 0.1  # Initial density of active sites
ACTIVATION_THRESHOLD = 0.8  # Threshold for a site to become active
REFRACTORY_PERIOD = 5  # Steps a site remains inactive after activation
NEIGHBORHOOD_SIZE = 1  # Moore neighborhood (1 means immediate neighbors)
BOUNDARY_CONDITIONS = "periodic"  # "periodic" or "fixed"

# --- Visualization Parameters ---
VISUALIZATION_FREQUENCY = 10  # Update plot every N steps
PLOT_STYLE = 'viridis'  # Colormap for visualization

class CellAutomaton1D:
    """
    A 1D Cellular Automaton simulating RUFT-like activation and refractory dynamics.
    """
    def __init__(self, size, initial_density, activation_threshold, refractory_period, neighborhood_size, boundary_conditions="periodic"):
        self.size = size
        self.initial_density = initial_density
        self.activation_threshold = activation_threshold
        self.refractory_period = refractory_period
        self.neighborhood_size = neighborhood_size
        self.boundary_conditions = boundary_conditions

        self.grid = np.random.rand(size) < initial_density
        self.refractory_timers = np.zeros(size, dtype=int)
        self.active_history = []
        self.entropy_history = []

    def get_neighbors_sum(self, index):
        """Calculates the sum of active neighbors for a given site."""
        s = 0
        for i in range(-self.neighborhood_size, self.neighborhood_size + 1):
            if i == 0:
                continue  # Skip the cell itself

            neighbor_idx = index + i
            if self.boundary_conditions == "periodic":
                neighbor_idx %= self.size
            elif self.boundary_conditions == "fixed":
                if not (0 <= neighbor_idx < self.size):
                    continue  # Skip out-of-bounds neighbors for fixed boundaries

            s += self.grid[neighbor_idx]
        return s

    def step(self):
        """Performs a single step of the simulation."""
        new_grid = self.grid.copy()
        newly_activated_count = 0

        # Update refractory timers
        self.refractory_timers[self.refractory_timers > 0] -= 1

        for i in range(self.size):
            if self.refractory_timers[i] > 0:
                new_grid[i] = False  # Site is in refractory period
                continue

            if self.grid[i]:  # Site is currently active
                # Rule: Active sites go into refractory period
                self.refractory_timers[i] = self.refractory_period
                new_grid[i] = False # Becomes inactive due to refractory, but was active this step for neighbors
                                    # This means neighbors see it as active for *this current step's* calculation
                                    # Then it becomes inactive for the *next step* due to refractory.
            else:  # Site is currently inactive and not in refractory
                # Rule: Inactive sites may become active based on neighbors
                # The neighbors' sum should use the state of the grid *before* this step's changes
                neighbors_sum = self.get_neighbors_sum(i)
                # Normalize by the number of possible neighbors
                max_neighbors = 2 * self.neighborhood_size
                activation_potential = neighbors_sum / max_neighbors if max_neighbors > 0 else 0

                if activation_potential >= self.activation_threshold:
                    new_grid[i] = True
                    self.refractory_timers[i] = self.refractory_period # Activate and immediately set refractory for next step
                    newly_activated_count +=1


        self.grid = new_grid
        self.active_history.append(np.sum(self.grid))
        self.entropy_history.append(self.calculate_shannon_entropy())
        return newly_activated_count

    def calculate_shannon_entropy(self):
        """Calculates Shannon entropy of the grid state."""
        # Consider only two states: active (1) and inactive (0)
        p_active = np.mean(self.grid)
        p_inactive = 1.0 - p_active

        if p_active == 0 or p_inactive == 0:
            return 0.0  # Entropy is 0 if all sites are in the same state

        entropy = - (p_active * np.log2(p_active) + p_inactive * np.log2(p_inactive))
        return entropy

    def run_simulation(self, time_steps, visualization_frequency=10, plot_style='viridis'):
        """Runs the simulation for a given number of time steps."""
        print(f"Starting 1D simulation: Size={self.size}, Steps={time_steps}, Density={self.initial_density:.2f}")
        plt.ion()
        fig, ax = plt.subplots(3, 1, figsize=(10, 12)) # Increased figure size for 3 plots

        # Plot 1: Grid state
        ax[0].set_title(f"1D Cellular Automaton (RUFT-like) - Step 0")
        ax[0].set_xlabel("Cell Index")
        ax[0].set_ylabel("State (Active=1, Inactive=0)")
        line, = ax[0].plot(self.grid, linestyle='-', marker='o', markersize=3)
        ax[0].set_ylim(-0.1, 1.1)
        ax[0].set_xlim(0, self.size)

        # Plot 2: Active sites count
        ax[1].set_title("Total Active Sites Over Time")
        ax[1].set_xlabel("Time Step")
        ax[1].set_ylabel("Number of Active Sites")
        ax[1].set_xlim(0, time_steps)
        active_line, = ax[1].plot([], [], color='blue')

        # Plot 3: Shannon Entropy
        ax[2].set_title("Shannon Entropy of Grid State Over Time")
        ax[2].set_xlabel("Time Step")
        ax[2].set_ylabel("Entropy (bits)")
        ax[2].set_xlim(0, time_steps)
        entropy_line, = ax[2].plot([], [], color='green')

        fig.tight_layout(pad=3.0) # Adjust layout

        start_time = time.time()
        for t in range(time_steps):
            newly_activated = self.step()
            if (t + 1) % visualization_frequency == 0 or t == 0:
                ax[0].set_title(f"1D Cellular Automaton (RUFT-like) - Step {t + 1}")
                line.set_ydata(self.grid)

                # Update active sites plot
                active_line.set_data(range(len(self.active_history)), self.active_history)
                ax[1].set_ylim(0, self.size * 1.1) # Dynamic y-limit

                # Update entropy plot
                entropy_line.set_data(range(len(self.entropy_history)), self.entropy_history)
                ax[2].set_ylim(0, 1.1) # Max Shannon entropy for binary system is 1

                plt.draw()
                plt.pause(0.01)
                print(f"Step {t + 1}/{time_steps} - Active: {self.active_history[-1]}, Newly Activated: {newly_activated}, Entropy: {self.entropy_history[-1]:.4f}")

        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
        print(f"Final active sites: {self.active_history[-1]}")
        print(f"Final entropy: {self.entropy_history[-1]:.4f}")

        plt.ioff()
        # Save final plot
        final_plot_filename = "1D_simulation_final_state.png"
        fig.savefig(final_plot_filename)
        print(f"Final state plot saved as {final_plot_filename}")

        # Save data
        np.save("1D_active_history.npy", np.array(self.active_history))
        np.save("1D_entropy_history.npy", np.array(self.entropy_history))
        print("Active history and entropy data saved.")

        plt.show()


if __name__ == "__main__":
    # --- Run the Simulation ---
    automaton = CellAutomaton1D(
        size=GRID_SIZE,
        initial_density=INITIAL_DENSITY,
        activation_threshold=ACTIVATION_THRESHOLD,
        refractory_period=REFRACTORY_PERIOD,
        neighborhood_size=NEIGHBORHOOD_SIZE,
        boundary_conditions=BOUNDARY_CONDITIONS
    )
    automaton.run_simulation(
        time_steps=TIME_STEPS,
        visualization_frequency=VISUALIZATION_FREQUENCY,
        plot_style=PLOT_STYLE
    )
    print("1D simulation run complete.")
    # To load data:
    # active_history = np.load("1D_active_history.npy")
    # entropy_history = np.load("1D_entropy_history.npy")
    # plt.plot(active_history)
    # plt.figure()
    # plt.plot(entropy_history)
    # plt.show()
