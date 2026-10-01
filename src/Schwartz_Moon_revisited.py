# ------------------------------------------------------------
# Schwartz and Moon (2001) model for simulating firm value.
# ------------------------------------------------------------
# The model is expressed inside the function simulate_firm_value.
# ---
# Simple bankruptcy handling (X < 0) and no seasonality adjusting.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import parameters as p  # Importing the parameters module in parameters.py
import os
import datetime
import pickle

def simulate_firm_value(gvkey, save_to_file=False):
    '''
    Inputs:
    - gvkey: The unique identifier for the firm.
    - save_to_file: Boolean indicating whether to save the simulation results to a file.
    
    Outputs:
    - V0: The expected net present value of the firm.
    '''
    # ------------------------------------------------------------
    # Model parameters. 
    # For explanation of the parameters, see the parameters.py file.
    # ------------------------------------------------------------
    firm_name = p.get_name(gvkey)  # Firm name

    R_0 = p.get_R_0(gvkey)  # Initial revenue in millions per quarter
    L_0 = p.get_L_0(gvkey)  # Initial Loss-carryforward in millions per quarter
    X_0 = p.get_X_0(gvkey)  # Initial cash balance in millions per quarter
    CapEx_Ratio_0 = p.get_Initial_CapEx_Ratio(gvkey)  # Initial CapEx ratio to revenue
    CapEx_Ratio_longterm = p.get_Long_term_CAPEX(gvkey)  # Long-term CapEx ratio to revenue
    Dep_Ratio = p.get_Depreciation_Ratio(gvkey)  # Depreciation ratio to PPE
    PPE_0 = p.get_PPE_0(gvkey) # Initial PPE (Property, Plant, Equipment) in millions

    mu_0 = p.get_mu_0()  # Initial expected growth rate per quarter
    sigma_0 = p.get_sigma_0()  # Initial revenue volatility per quarter
    eta_0 = p.get_eta_0(gvkey)  # Initial volatility of expected growth rate
    rho_R_mu = p.get_rho_R_mu() # Correlation between revenue and growth rate

    mu_mean = p.get_mu_mean()  # Mean-reversion level for growth rate
    sigma_mean = p.get_sigma_mean()  # Mean-reversion level for volatility
    
    taxrate = p.get_taxrate(gvkey)  # Corporate tax rate
    r_f = p.get_r_f()  # Risk-free rate

    kappa_mu = 0.09 # p.get_kappa_mu()  # Mean-reversion speed for expected growth rate
    kappa_sigma = p.get_kappa_sigma()  # Mean-reversion speed for volatility
    kappa_eta = p.get_kappa_eta()  # Mean-reversion speed for expected growth rate volatility
    kappa_gamma = p.get_kappa_gamma()  # Mean-reversion speed for gamma (cost ratio to revenue)
    kappa_phi = p.get_kappa_phi()  # Mean-reversion speed for phi (volatility of cost ratio)
    kappa_capex = p.get_kappa_capex()  # Mean-reversion speed for CapEx ratio

    gamma_0 = p.get_gamma_0(gvkey)  # Initial cost ratio to revenue
    gamma_mean = p.get_gamma_mean(gvkey)  # Mean-reversion level for cost ratio to revenue
    phi_0 = p.get_phi_0()  # Initial volatility of cost ratio to revenue
    phi_mean = p.get_phi_mean()  # Mean-reversion level for volatility of cost ratio to revenue

    lambda_R = p.get_lambda_R()  # Market price of risk for the revenue factor
    lambda_mu = p.get_lambda_mu()  # Market price of risk for the expected rate of growth in revenues factor
    lambda_gamma = p.get_lambda_gamma()  # Market price of risk for the cost ratio to revenue factor

    T = p.get_T()  # Time horizon in years
    dt = p.get_dt() # Time step
    M = p.get_M() # Exit multiple
    simulations = p.get_simulations()  # Number of Monte Carlo runs

    num_steps = (T * 4) + 1 # Quarters in T years + initial step

    np.random.seed(gvkey) # Seed random number generator for reproducibility

    # ------------------------------------------------------------
    # Allocate arrays for simulation results
    # ------------------------------------------------------------
    shape = (simulations, num_steps)  # Shape of the arrays

    R = np.zeros(shape) # Revenue trajectories
    Cost = np.zeros(shape)  # Cost trajectories
    L = np.zeros(shape)  # Loss carryforward trajectories
    X = np.zeros(shape) # Cash balance trajectories
    CapEx_ratio = np.zeros(num_steps)  # CapEx ratio trajectories
    CapEx = np.zeros(shape)  # Capital Expenditures trajectories   
    Dep = np.zeros(shape)  # Depreciation trajectories
    PPE = np.zeros(shape)  # Property, Plant, and Equipment trajectories
    EBITDA = np.zeros(shape)  # Earnings before interest and taxes
    Tax = np.zeros(shape)  # Tax trajectories
    NOPAT = np.zeros(shape)  # Net Operating Profit After Tax trajectories
    mu = np.zeros(shape)  # Growth rate trajectories
    gamma = np.zeros(shape)  # Cost ratio trajectories
    phi = np.zeros(num_steps)  # Volatility of cost ratio trajectories
    sigma = np.zeros(num_steps)  # Volatility of revenue trajectories
    eta = np.zeros(num_steps)  # Volatility of expected growth rate trajectories
    bankruptcy = np.zeros(shape, dtype=bool) # Bankruptcy indicator

    # ------------------------------------------------------------
    # Initial conditions (t=0)
    # ------------------------------------------------------------
    R[:, 0] = R_0  # Initial revenue
    X[:, 0] = X_0  # Initial cash balance
    Cost[:, 0] = gamma_0 * R_0  # Initial cost
    CapEx[:, 0] = CapEx_Ratio_0 * R_0  # Initial CapEx
    CapEx_ratio[0] = CapEx_Ratio_0  # Initial CapEx ratio
    PPE[:, 0] = PPE_0  # Initial PPE
    Dep[:, 0] = np.nan  # Initial depreciation (not used in the first step)
    Tax[:, 0] = np.nan  # Initial tax (not used in the first step)
    NOPAT[:, 0] = np.nan  # Initial NOPAT (not used in the first step)
    mu[:, 0] = mu_0  # Initial expected growth rate
    gamma[:, 0] = gamma_0  # Initial cost ratio to revenue
    phi[0] = phi_0  # Initial volatility of cost ratio to revenue
    sigma[0] = sigma_0  # Initial revenue volatility
    eta[0] = eta_0  # Initial volatility of expected growth rate
    L[:, 0] = L_0  # Initial loss carryforward
    bankruptcy[:, 0] = False  # Initial bankruptcy indicator

    # ------------------------------------------------------------
    # Random shocks
    # ------------------------------------------------------------
    Z_R = np.random.randn(simulations, num_steps)  # Standard normal noise for revenue
    Z_mu = rho_R_mu * Z_R + np.sqrt(1 - rho_R_mu**2) * np.random.randn(simulations, num_steps)  # Correlated noise for growth
    Z_gamma = np.random.randn(simulations, num_steps)  # Standard normal noise for gamma (cost ratio to revenue)


    # ------------------------------------------------------------
    # Forward Monte‑Carlo simulation
    # ------------------------------------------------------------
    for t in range(1, num_steps):  # Start from t=1 to num_steps-1

        # Only update non-bankrupt firms
        active_firms = ~bankruptcy[:, t-1]  # Firms that haven't gone bankrupt in the previous step
        
        # Update revenue using stochastic process
        R[:, t] = R[:, t-1] * np.exp(
                (mu[:, t-1] - lambda_R * sigma[t-1] - 0.5 * sigma[t-1]**2) * dt + sigma[t-1] * np.sqrt(dt) * Z_R[:, t] # eq28, SchosserStröbele2019
            )
        
        # Update growth rate with mean-reversion
        mu[:, t] = np.exp(-kappa_mu * dt) * mu[:, t-1] + (1 - np.exp(-kappa_mu * dt)) * (mu_mean - ((lambda_mu*eta[t-1])/(kappa_mu))) + np.sqrt((1 - np.exp(-2*kappa_mu*dt))/(2*kappa_mu)) * eta[t-1] * Z_mu[:, t] # eq29 i SchosserStröbele2019

        # Gamma (cost ratio to revenue)
        gamma[:, t] = np.exp(-kappa_gamma*dt) * gamma[:, t-1] + (1 - np.exp(-kappa_gamma*dt)) * (gamma_mean - ((lambda_gamma * phi[t-1])/(kappa_gamma))) + np.sqrt((1 - np.exp(-2*kappa_gamma*dt))/(2*kappa_gamma)) * phi[t-1] * Z_gamma[:, t] 


        # Sigma (volatility in revenue)
        sigma[t] = sigma[0] * np.exp(-kappa_sigma * t) + sigma_mean * (1 - np.exp(-kappa_sigma * t)) # eq19 SM2000 og eq32 SchosserStröbele2019
        
        # Update expected growth rate volatility
        eta[t] = eta[0] * np.exp(-kappa_eta * t) # eq20, SM2000 og schosser strobele eq33

        # Phi
        phi[t] = np.exp(-kappa_phi * t) * phi[0] + (1 - np.exp(-kappa_phi * t)) * phi_mean # eq34 in SchosserStrobele and eq30 in SM2001

        # CapEx ratio, TODO: discuss model improvement by NotlandTaraldsen
        CapEx_ratio[t] = CapEx_ratio[0] * np.exp(-kappa_capex * t) + CapEx_Ratio_longterm * (1 - np.exp(-kappa_capex * t)) # Introduced here by NotlandTaraldsen, a mean reverting capex ratio.

        # Cost
        Cost[:, t] = gamma[:, t] * R[:, t] # We use total cost ratio, gamma er excluding depreciation og amortization, but including interest expense. TODO: Discuss implications in thesis.

        # Depreciation
        Dep[:, t] = Dep_Ratio * PPE[:, t-1]  # Depreciation is calculated based on the previous period's PPE.

        # Update CapEx
        CapEx[:, t] = CapEx_ratio[t] * R[:, t]  # CapEx is calculated based on the current period's revenue.

        # PPE
        PPE[:, t] = PPE[:, t-1] - Dep[:, t] + CapEx[:, t]  # Update PPE based on depreciation and CapEx.

        # Compute Tax in absolute value (eq14 in SchosserStrobele)
        # Tax is computed quarterly, and determined using loss-carryforward from company data. 
        taxable_income = R[:, t] - Cost[:, t] - Dep[:, t] - L[:, t-1]
        Tax[:, t] = np.where(taxable_income <= 0, 0, taxable_income * taxrate)
        # TODO: Discuss tax rate and the choice on quarterly tax in the thesis. 
        
        #  Compute Net Operating Profit After Tax (NOPAT) (eq14 in SchosserStrobele)
        NOPAT[:, t] = R[:, t] - Cost[:, t] - Dep[:, t] - Tax[:, t]

        # Compute Loss Carryforward
        used_loss = NOPAT[:, t] + Tax[:, t]
        L[:, t] = np.where(L[:, t-1] > used_loss, L[:, t-1] - used_loss, 0)  # Loss carryforward is reduced by the taxable income, but not below zero.

        # Update cash balance
        X[:, t] = X[:, t-1] + (r_f * X[:, t-1] + NOPAT[:, t] + Dep[:, t] - CapEx[:, t]) * dt
        
        # Check for bankruptcy
        bankruptcy[active_firms, t] = X[active_firms, t] < 0  # Mark bankruptcy if cash is non-positive
        bankruptcy[bankruptcy[:, t], t:] = True  # Mark future time steps as bankrupt
        
        # If a company goes bankrupt, set all future values to zero
        X[bankruptcy[:, t], t:] = 0
        R[bankruptcy[:, t], t:] = 0
        L[bankruptcy[:, t], t:] = 0
        EBITDA[bankruptcy[:, t], t:] = 0


    # Compute Discounted Expected Value of the Firm
    terminal_value = M * (R[:, -1] - Cost[:, -1])
    # Adjust for bankrupt firms (ensures terminal value is also zero if bankrupt)
    terminal_value[bankruptcy[:, -1]] = 0  # No terminal value if bankrupt
    # Compute Expected Firm Value (DCF approach)
    V0 = np.mean((X[:, -1] + terminal_value) * np.exp(-r_f * T))
    num_bankruptcies = np.sum(bankruptcy[:, -1])  # Count bankruptcies at the end of the simulation

    # ------------------------------------------------------------
    # Save simulation results to file
    # ------------------------------------------------------------
    if save_to_file:
        # Create a timestamp for model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a dictionary with all relevant data
        results = {
            "timestamp": timestamp,
            "gvkey": gvkey,
            "name": firm_name,
            "parameters": {
                "R_0": R_0,
                "L_0": L_0,
                "X_0": X_0,
                "CapEx_Ratio_0": CapEx_Ratio_0,
                "CapEx_Ratio_longterm": CapEx_Ratio_longterm,
                "Dep_Ratio": Dep_Ratio,
                "PPE_0": PPE_0,
                "mu_0": mu_0,
                "sigma_0": sigma_0,
                "eta_0": eta_0,
                "rho_R_mu": rho_R_mu,
                "mu_mean": mu_mean,
                "sigma_mean": sigma_mean,
                "taxrate": taxrate,
                "r_f": r_f,
                "kappa_mu": kappa_mu,
                "kappa_sigma": kappa_sigma,
                "kappa_eta": kappa_eta,
                "kappa_gamma": kappa_gamma,
                "kappa_phi": kappa_phi,
                "kappa_capex": kappa_capex,
                "gamma_0": gamma_0,
                "gamma_mean": gamma_mean,
                "phi_0": phi_0,
                "phi_mean": phi_mean,
                "lambda_R": lambda_R,
                "lambda_mu": lambda_mu,
                "lambda_gamma": lambda_gamma,
                "T": T,
                "dt": dt,
                "M": M,
                "simulations": simulations,
            },
            "results": {
                "R": R,
                "X": X,
                "Cost": Cost,
                "CapEx": CapEx,
                "CapEx_ratio": CapEx_ratio,
                "Dep": Dep,
                "PPE": PPE,
                "Tax": Tax,
                "mu": mu,
                "gamma": gamma,
                "phi": phi,
                "sigma": sigma,
                "eta": eta,
                "L": L,
                "bankruptcy": bankruptcy,
                "EBITDA": EBITDA,
                "terminal_value": terminal_value,
                "V0": V0,
                "num_bankruptcies": num_bankruptcies,
            }
        }

        ### Commented out add to all sims, only keep latest sim. ###

        # filename_complete = f"{gvkey}_sim_results_{timestamp}.pkl"
        filename_latest_sim = f"v0_{gvkey}_latest_sim_results.pkl"

        # Save to disk
        # output_dir_all = "simulation_outputs_all"
        output_dir_latest = "simulation_outputs_latest"  # This directory is not saved to git, large data files.
        # os.makedirs(output_dir_all, exist_ok=True)
        os.makedirs(output_dir_latest, exist_ok=True)

        # filepath_all = os.path.join(output_dir_all, filename_complete)
        # with open(filepath_all, "wb") as f:
            # pickle.dump(results, f)

        filepath_latest = os.path.join(output_dir_latest, filename_latest_sim)
        with open(filepath_latest, "wb") as f:
            pickle.dump(results, f)

        # print("Simulation results saved to:", filepath_all)
        print("Latest simulation results saved to:", filepath_latest)

    # returns only the expected net present value of the firm.
    return V0



if __name__ == "__main__":
    # Simulate firm value for each company in the list
    for gvkey in p.COMPANY_LIST:
        simulate_firm_value(gvkey, save_to_file=True)
