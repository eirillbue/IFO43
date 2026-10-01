# sensitivity_analysis.py
# This script runs sensitivity analysis on the firm's value simulation model

import numpy as np
import pandas as pd
from NT_2025_model import simulate_firm_value_sensitivity
import parameters as p

def run_sensitivity_analysis(gvkey):
    # --- CONFIGURATION ---
    np.random.seed(seed=gvkey) # Set seed based on gvkey for reproducibility

    # --- Generate shocks ONCE ---
    simulations = p.get_simulations() # should be 1 000 000
    num_steps = p.get_num_steps() # should be 101

    Z_R = np.random.randn(simulations, num_steps)
    Z_mu = np.random.randn(simulations, num_steps)
    Z_gamma = np.random.randn(simulations, num_steps)

    # --- Base values from parameters ---
    base_params = {
        "r_f": p.get_r_f(),
        "mu_0": p.get_mu_0(),
        "mu_mean": p.get_mu_mean(),
        "sigma_0": p.get_sigma_0(),
        "sigma_mean": p.get_sigma_mean(),
        "eta_0": p.get_eta_0(gvkey),
        "phi_0": p.get_phi_0(),
        "phi_mean": p.get_phi_mean(),
        "lambda_R": p.get_lambda_R(),
        "lambda_mu": p.get_lambda_mu(),
        "lambda_gamma": p.get_lambda_gamma(),
        "kappa_mu": 0.09,
        "kappa_eta": p.get_kappa_eta(),
        "kappa_sigma": p.get_kappa_sigma(),
        "kappa_gamma": p.get_kappa_gamma(),
        "kappa_phi": p.get_kappa_phi(),
        "kappa_capex": p.get_kappa_capex(),
        "financing_cost": p.get_financing_cost(),
        "financing_grid": p.get_financing_grid(gvkey),
        "M": p.get_exit_multiple()
    }

    # --- Run base case ---
    print("Running base case simulation...")
    V0_base, base_bankruptcies = simulate_firm_value_sensitivity(
        gvkey,
        Z_R=Z_R,
        Z_mu=Z_mu,
        Z_gamma=Z_gamma,
        override_params={}  # no overrides
    )
    print(f"Base case: V0 = {V0_base:.2f}, Bankruptcies = {base_bankruptcies}")


    # Define 10% increase for each parameter
    parameter_changes = {key: (lambda val: val * 1.1) for key in base_params}

    # --- Store results ---
    results = []

    for param, modifier in parameter_changes.items():
        print(f"Running sensitivity analysis for parameter: {param}...")

        overrides = {}
        base_val = base_params[param]
        new_val = modifier(base_val)

        if param == "financing_grid":
            overrides[param] = new_val  # Apply elementwise 10% increase
        else:
            overrides[param] = new_val

        # Run simulation with overridden parameter
        V0, num_bankruptcies = simulate_firm_value_sensitivity(
            gvkey,
            Z_R=Z_R,
            Z_mu=Z_mu,
            Z_gamma=Z_gamma,
            override_params=overrides
        )

        percentage_change = ((V0 - V0_base) / V0_base) * 100

        print(f"Finished {param}: V0 = {V0:.2f}, Bankruptcies = {num_bankruptcies}")


        results.append({
            "parameter": param,
            "base_value": base_val,
            "new_value": new_val,
            "V0": V0,
            "num_bankruptcies": num_bankruptcies,
            "V0_base": V0_base,
            "percentage_change_from_base": percentage_change
        })

    # --- Output to DataFrame and file ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"results/{gvkey}_sensitivity_results.csv", index=False)
    print(df_results)


if __name__ == "__main__":
    for gvkey in p.COMPANY_LIST:
        # Run sensitivity analysis for each gvkey in the list
        print(f"Running sensitivity analysis for gvkey: {gvkey}")
        run_sensitivity_analysis(gvkey)
        
    