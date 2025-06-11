import numpy as np
import h5py
import os
from dataclasses import dataclass, asdict
from astropy import units as u
from astropy.constants import c
import time
import traceback

# --- Part 1: Parameter Setup & Unit Conversion ---
@dataclass
class SimParameters:
    """Manages physical parameters and converts them to a unitless system for the solver."""
    # ... (This class remains unchanged from v2.1)
    L: u.Quantity = 100 * u.m
    t_final: u.Quantity = 2 * u.s
    c_scaled: u.Quantity = 300 * u.m / u.s
    alpha: u.Quantity = 0.1 * u.m**2 / u.kg
    beta: u.Quantity = 1e-5 * u.m**3 / u.kg
    kappa: u.Quantity = 0.2 / (u.m * u.s**2)
    rho_0: u.Quantity = 1e5 * u.kg / u.m**3
    omega: u.Quantity = 100 / u.s
    k: u.Quantity = 0.1 / u.m
    Nx: int = 200
    Nt: int = 4000
    save_interval: int = 50

    def get_unitless_params(self):
        """Converts all physical parameters into a dimensionless dictionary for the solver."""
        # ... (This method remains unchanged from v2.1)
        print("--- Converting Physical Parameters to Unitless System ---")
        base_m, base_s, base_kg = 1.0 * u.m, 1.0 * u.s, 1.0 * u.kg
        p = {
            'L': (self.L / base_m).value, 't_final': (self.t_final / base_s).value,
            'c_scaled': (self.c_scaled / (base_m / base_s)).value,
            'alpha': (self.alpha / (base_m**2 / base_kg)).value,
            'beta': (self.beta / (base_m**3 / base_kg)).value,
            'kappa': (self.kappa / (1 / (base_m * u.s**2))).value,
            'rho_0': (self.rho_0 / (base_kg / base_m**3)).value,
            'omega': (self.omega / (1 / base_s)).value,
            'k': (self.k / (1 / base_m)).value,
            'Nx': self.Nx, 'Nt': self.Nt, 'save_interval': self.save_interval
        }
        p['dx'] = p['L'] / (p['Nx'] - 1)
        p['dt'] = p['t_final'] / p['Nt']
        print("--- Unitless Parameters for Solver ---")
        for key, val in p.items():
            print(f"{key:<12}: {val:.3e}")
        print("------------------------------------")
        return p

# --- Part 2: The Unit-less Numerical Solver ---
def compute_IP(rho, grad_rho, t_step, p):
    """Calculates the Intrinsic Potential using unitless parameters."""
    x = np.arange(p['Nx']) * p['dx']
    phi = p['omega'] * (t_step * p['dt']) + p['k'] * x
    return p['alpha'] * np.abs(grad_rho) * np.exp(-p['beta'] * rho / p['rho_0']) * np.cos(phi)

def run_simulation(p, output_filename="output_1d.h5"):
    """
    Main function to run the 1D RUFT simulation using a unitless parameter dictionary 'p'.
    """
    print("--- RUFT 1D Simulation (v2.2) Initializing ---")

    # --- ENHANCEMENT: Robust Path Handling ---
    # This section ensures the output file is always saved in the same directory as the script.
    try:
        # Get the absolute path of the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the output path relative to the script's location
        output_path = os.path.join(script_dir, output_filename)
        # The directory should already exist if the script is there, but this is a failsafe.
        os.makedirs(script_dir, exist_ok=True)
    except NameError:
        # __file__ is not defined in some environments (e.g., interactive interpreters)
        # Fallback to current working directory
        print("Warning: Could not determine script directory. Saving to current working directory.")
        output_path = output_filename

    # --- Setup Grid and Initial Conditions ---
    x = np.linspace(0, p['L'], p['Nx'])
    U = np.zeros((p['Nt'], p['Nx']))
    rho = np.zeros((p['Nt'], p['Nx']))
    rho[0, :] = p['rho_0'] * 0.1 * np.exp(-((x - p['L']/2)**2) / (p['L']/10)**2)
    U[0, :] = rho[0, :] / (p['c_scaled']**2)
    U[1, :] = U[0, :] # Initial condition for the second time step for the leapfrog scheme

    # --- ENHANCEMENT: Comprehensive Try/Except block for HDF5 and Solver ---
    try:
        with h5py.File(output_path, 'w') as f:
            print(f"HDF5 output file will be saved to: {output_path}")
            param_group = f.create_group("SIMULATION_PARAMETERS")
            param_group.attrs.update(p)
            mesh_group = f.create_group("MESH_DATA")
            mesh_group.create_dataset("x_grid", data=x)
            timesteps_group = f.create_group("TIMESTEPS")

            start_time = time.time()
            # --- Main Solver Loop ---
            for t_step in range(1, p['Nt'] - 1):
                # Using t_step for current values to calculate t_step+1
                lap_U = np.zeros(p['Nx'])
                lap_U[1:-1] = (U[t_step, 2:] - 2*U[t_step, 1:-1] + U[t_step, :-2]) / p['dx']**2

                grad_rho = np.zeros(p['Nx'])
                grad_rho[1:-1] = (rho[t_step, 2:] - rho[t_step, :-2]) / (2 * p['dx'])

                ip = compute_IP(rho[t_step, :], grad_rho, t_step, p)
                source = p['kappa'] * ip * np.sin(p['omega'] * (t_step * p['dt']) + p['k'] * x)

                # Leapfrog method for the wave equation on U
                U_next = (2 * U[t_step, :] - U[t_step-1, :] +
                          p['dt']**2 * (p['c_scaled']**2 * lap_U + source))

                # Forward Euler for rho evolution
                rho_next = rho[t_step, :] + p['dt'] * ip

                # --- Stability Engineering: Clamp values to prevent explosion ---
                U[t_step+1, :] = np.clip(U_next, -1e12, 1e12)
                rho[t_step+1, :] = np.clip(rho_next, 0, 1e7) # Mass density cannot be negative

                # --- Data Saving & Feedback ---
                if t_step % p['save_interval'] == 0:
                    step_group = timesteps_group.create_group(str(t_step + 1))
                    fields_group = step_group.create_group("fields")
                    fields_group.create_dataset("rho", data=rho[t_step+1, :])
                    fields_group.create_dataset("U", data=U[t_step+1, :])

                    stats_group = step_group.create_group("stats")
                    stats_group.attrs["sim_time_s"] = (t_step + 1) * p['dt']

                    elapsed = time.time() - start_time
                    print(f"Step: {t_step+1}/{p['Nt']}, Time: {(t_step+1)*p['dt']:.3f}s, Max U: {np.max(U[t_step+1, :]):.2e}, Wall Time: {elapsed:.2f}s")

        print("--- RUFT 1D Simulation (v2.2) Finished ---")

    except IOError as e:
        print(f"FATAL ERROR: Could not write to HDF5 file at '{output_path}'.")
        print(f"System Error: {e}")
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred during the simulation.")
        traceback.print_exc() # Print the full traceback for detailed debugging
        print("Simulation halted.")


if __name__ == "__main__":
    physical_params = SimParameters()
    unitless_solver_params = physical_params.get_unitless_params()
    run_simulation(unitless_solver_params)
