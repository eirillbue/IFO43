# ------------------------------------------------------------
# NT_2025_model.py (latest version of the model, baseline valuation model)
# ------------------------------------------------------------
# The model is expressed inside the function simulate_firm_value.
# This model is part of a master's thesis project at the Norwegian University of Science and Technology (NTNU) in Norway.
# ---
# Authors: 
# Per Inge Notland
# David Taraldsen
# ---
# Date: 05.06.2025
# ------------------------------------------------------------

import numpy as np
import parameters  as p  # Importing the parameters methods from the parameters.py file
import os
import datetime
import pickle

def basis(x_cash, rev):
    """
    simple polynomial basis in (X,R)
    for the cash flow process
    x_cash: cash balance
    rev: revenue
    """
    return np.column_stack([np.ones_like(x_cash),
                            x_cash,
                            rev,
                            x_cash**2])

def simulate_firm_value(gvkey, save_to_file=False):
    '''
    Inputs:
    - gvkey: The unique identifier for the firm.
    - save_to_file: Boolean indicating whether to save the simulation results to a file.
    
    Outputs:
    - V0_LSM: The expected net present value of the firm at time t=0.
    '''
    # ------------------------------------------------------------
    # Model parameters
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
    M = p.get_exit_multiple() # Exit multiple
    simulations = p.get_simulations()  # Number of Monte Carlo runs
    seasonal_factors = p.get_seasonal_factors(gvkey)  # Seasonal factors for revenue

    financing_cost = p.get_financing_cost() # 0.02  # Cost of issuing new equity (2% of cash injection) 
    financing_grid = p.get_financing_grid(gvkey) # np.array([0.0, 5.0, 10.0, 20.0, 40.0])  # Cash injection grid (in millions EUR).
    C_max = p.get_C_max(gvkey)

    num_steps = p.get_num_steps()

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
    financing_matrix = np.zeros(shape)  # Track financing amounts per time step
    beta_matrix = np.zeros((num_steps, 4))  # Store 4 regression coefficients for each timestep


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
    mu[:, 0] = mu_0 # Initial growth rate 
    gamma[:, 0] = gamma_0  # Initial cost ratio to revenue
    phi[0] = phi_0  # Initial volatility of cost ratio to revenue
    sigma[0] = sigma_0  # Initial revenue volatility
    eta[0] = eta_0  # Initial expected growth rate volatility
    L[:, 0] = L_0 # Initial loss carry-forward
    bankruptcy[:, 0] = False # Initial bankruptcy indicator
    
    # ------------------------------------------------------------
    # Random shocks
    # ------------------------------------------------------------
    Z_R = np.random.randn(simulations, num_steps)  # Standard normal noise for revenue
    Z_mu = rho_R_mu * Z_R + np.sqrt(1 - rho_R_mu**2) * np.random.randn(simulations, num_steps)  # Correlated noise for growth
    Z_gamma = np.random.randn(simulations, num_steps)  # Standard normal noise for gamma (cost ratio to revenue)

    # ------------------------------------------------------------
    # Forward Monte‑Carlo simulation
    # ------------------------------------------------------------
    for t in range(1, num_steps):

        # Get current quarter for seasonality adjustment
        quarter = (t % 4) if (t % 4) != 0 else 4
        seasonal_factor = seasonal_factors.get(quarter)
        
        # Update revenue using stochastic process and seasonal factor
        R[:, t] = R[:, t-1] * np.exp(
                (mu[:, t-1] - lambda_R * sigma[t-1] - 0.5 * sigma[t-1]**2) * dt + sigma[t-1] * np.sqrt(dt) * Z_R[:, t] # eq28, SchosserStröbele
            ) * seasonal_factor
        
        # Update growth rate with mean-reversion
        mu[:, t] = np.exp(-kappa_mu * dt) * mu[:, t-1] + (1 - np.exp(-kappa_mu * dt)) * (mu_mean - ((lambda_mu*eta[t-1])/(kappa_mu))) + np.sqrt((1 - np.exp(-2*kappa_mu*dt))/(2*kappa_mu)) * eta[t-1] * Z_mu[:, t] # eq29 i SchosserStröbele

        # Gamma (cost ratio to revenue)
        gamma[:, t] = np.exp(-kappa_gamma*dt) * gamma[:, t-1] + (1 - np.exp(-kappa_gamma*dt)) * (gamma_mean - ((lambda_gamma * phi[t-1])/(kappa_gamma))) + np.sqrt((1 - np.exp(-2*kappa_gamma*dt))/(2*kappa_gamma)) * phi[t-1] * Z_gamma[:, t] 


        # Sigma (volatility in revenue)
        sigma[t] = sigma[0] * np.exp(-kappa_sigma * t) + sigma_mean * (1 - np.exp(-kappa_sigma * t)) # eq19 SM2000 og eq32 SchosserStröbele
        
        # Update expected growth rate volatility
        eta[t] = eta[0] * np.exp(-kappa_eta * t) # eq20, SM2000 og schosser strobele eq33

        # Phi
        phi[t] = np.exp(-kappa_phi * t) * phi[0] + (1 - np.exp(-kappa_phi * t)) * phi_mean # eq34 in SchosserStrobele and eq30 in SM2001

        # CapEx ratio 
        CapEx_ratio[t] = CapEx_ratio[0] * np.exp(-kappa_capex * t) + CapEx_Ratio_longterm * (1 - np.exp(-kappa_capex * t)) # Mean reverting Capex ratio.

        # Cost 
        Cost[:, t] = gamma[:, t] * R[:, t] # We use total cost ratio, gamma is excluding depreciation og amortization, but including interest expense.

        # Depreciation
        Dep[:, t] = Dep_Ratio * PPE[:, t-1]

        # Update CapEx
        CapEx[:, t] = CapEx_ratio[t] * R[:, t]

        # PPE
        PPE[:, t] = PPE[:, t-1] - Dep[:, t] + CapEx[:, t] 

        # Compute Tax in absolute value (eq14 in SchosserStrobele)
        taxable_income = R[:, t] - Cost[:, t] - Dep[:, t] - L[:, t-1]
        Tax[:, t] = np.where(taxable_income <= 0, 0, taxable_income * taxrate)

        # Update Net Operating Profit After Tax (NOPAT)
        NOPAT[:, t] = R[:, t] - Cost[:, t] - Dep[:, t] - Tax[:, t]

        # Compute Loss Carryforward
        used_loss = NOPAT[:, t] + Tax[:, t]
        L[:, t] = np.where(L[:, t-1] > used_loss, L[:, t-1] - used_loss, 0)

        # Update cash balance
        X[:, t] = X[:, t-1] + (r_f * X[:, t-1] + NOPAT[:, t] + Dep[:, t] - CapEx[:, t]) * dt


    

    # ------------------------------------------------------------
    # Longstaff‑Schwartz backward sweep with financing option
    # ------------------------------------------------------------
    discount = np.exp(-r_f*dt)  # Discount factor for cash flows

    NAV_proxy = PPE[:,-1]

    terminal = X[:, -1] + M * NAV_proxy

    V            = np.zeros_like(X)  # Value function
    V[:,-1]      = terminal  # Terminal value at maturity
    
    abandonment_value = 0.0  # Maybe later include a salvage value here, following Gamba
    bankrupt_now = np.zeros(simulations, dtype=bool)  # Track bankruptcy at time t
    bankruptcy = np.zeros((simulations, num_steps), dtype=bool)  # Track bankruptcy for all time steps

    r_squared = np.full(num_steps, np.nan)
    adj_r_squared = np.full(num_steps, np.nan)
    rmse = np.full(num_steps, np.nan)

    for t in range(num_steps - 2, -1, -1):  # Backward iteration over time steps
        cont_disc = discount * V[:, t + 1]  # Discounted continuation value

        # Identify paths that are not bankrupt and below the cash cutoff
        # The following lines are used to determine the paths that will be used for regression
        # Right now, the 20% firms with lowest cash, that are not bankrupt, are used for regression.
        percentile_cutoff = 20  # Percentile cutoff for cash balance
        not_bankrupt = ~bankruptcy[:, t+1]  # Paths that are not bankrupt
        nonbankrupt_cash = X[:, t][not_bankrupt]  # Cash balance of non-bankrupt paths
        if nonbankrupt_cash.size == 0:
            # Fallback: use all firms in the simulation
            valid_regression_paths = np.zeros(simulations, dtype=bool)
        else:
            cash_cutoff = np.percentile(nonbankrupt_cash, percentile_cutoff)
            valid_regression_paths = (X[:, t] <= cash_cutoff) & not_bankrupt

        min_paths = 10  # Minimum number of paths for regression
        if np.sum(valid_regression_paths) >= min_paths:
            B = basis(X[valid_regression_paths, t], R[valid_regression_paths, t])  # Basis for regression
            Y = cont_disc[valid_regression_paths]  # Discounted continuation value for valid paths
        else:  # If not enough paths, use all paths for regression
            B = basis(X[:, t], R[:, t])
            Y = cont_disc

        beta, *_ = np.linalg.lstsq(B, Y, rcond=None)  # Fit regression model (beta_0 + beta_1*X + beta_2*R + beta_3*X^2)
        beta_matrix[t, :] = beta  # Store regression coefficients
        
        C_hat_0 = basis(X[:, t], R[:, t]) @ beta  # Predicted continuation value given current cash and revenue
        
        # Compute regression diagnostics
        Y_pred = B @ beta
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        n = len(Y)
        p1 = B.shape[1]

        # R²
        r_squared[t] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Adjusted R²
        if ss_tot > 0 and n > p1:
            adj_r_squared[t] = 1 - (1 - r_squared[t]) * (n - 1) / (n - p1 - 1)

        # RMSE
        rmse[t] = np.sqrt(np.mean((Y - Y_pred) ** 2))


        best_val = C_hat_0.copy()  # Initialize best value with predicted continuation value
        best_f = np.zeros(simulations)  # Initialize best financing choice

        for f in financing_grid[1:]:  # iterate over financing amounts (excluding 0)
            X_tmp = X[:, t] + f  # Cash balance after financing
            C_tmp = basis(X_tmp, R[:, t]) @ beta  # Predicted continuation value after financing
            val = -financing_cost * f + C_tmp  # Value after financing
            can_finance = (X[:, t] < C_max) & (val > 0)  # Paths that can finance and have positive value
            mask = (val > best_val) & can_finance # Identify paths where financing is better than current value
            best_val[mask] = val[mask]  # Update best value
            best_f[mask] = f  # Update financing choice
        
        if t == 0:
            best_f[:] = 0
            best_val[:] = C_hat_0  # Set financing choice to 0 for t=0

        # Identify bankrupt paths: cash < 0 and no financing chosen or negative value
        bankrupt_now = ((X[:, t] < 0) & (best_f == 0)) | (best_val < 0)  # Paths that are bankrupt now

        # Record bankruptcies for analysis
        bankruptcy[bankrupt_now, t:] = True

        # Set value to abandonment value for bankrupt paths
        best_val[bankrupt_now] = abandonment_value

        financing_matrix[:, t] = best_f  # Store chosen financing amount at time t
        financing_matrix[bankrupt_now, t:] = 0.0

        # Adjust cash flow for financing, but not for bankrupt paths
        cash_flow = -financing_cost * best_f * (~bankrupt_now)
        V[:, t] = cash_flow + best_val
        V[:, t] = np.maximum(V[:, t], 0.0)

    total_discounted_financing = np.sum(financing_matrix * np.exp(-r_f * np.arange(num_steps) * dt), axis=1)

    V0_LSM = np.mean(V[:,0]- total_discounted_financing) - X_0
    
    # V0_LSM = np.mean(V[:,0]) - X_0  # Hadde i prinsippet ikke trengt å ta snittet for alle burde være like når t=0. pga lik cash og revenue.
    # v7: trekke fra initial cash balance, slik at vi får enterprise value.
    num_bankruptcies = np.sum(np.any(bankruptcy, axis=1)) # Count bankruptcies across all time steps

    # ------------------------------------------------------------
    # Save simulation results to file
    # ------------------------------------------------------------
    if save_to_file:
        # Create a timestamped filename
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
                "financing_cost": financing_cost,
                "financing_grid": financing_grid,
                "seasonal_factors": seasonal_factors,
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
                "terminal_value": terminal,
                "V0": V0_LSM,
                "num_bankruptcies": num_bankruptcies,
                "financing": financing_matrix,
                "betas": beta_matrix,
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "rmse": rmse,
            }
        }

        ### Commented out add to all sims, only keep latest sim. ###

        # filename_complete = f"{gvkey}_sim_results_{timestamp}.pkl"
        filename_latest_sim = f"v8_{gvkey}_latest_sim_results.pkl"

        # Save to disk
        # output_dir_all = "simulation_outputs_all"
        output_dir_latest = "simulation_outputs_latest"
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
    return V0_LSM



# ------------------------------------------------------------
# Sensitivity analysis function
# ------------------------------------------------------------

def simulate_firm_value_sensitivity(gvkey, Z_R=None, Z_mu=None, Z_gamma=None, override_params=None, save_to_file=False):
    '''
    Inputs:
    - gvkey: The unique identifier for the firm.
    - save_to_file: Boolean indicating whether to save the simulation results to a file.
    
    Outputs:
    - V0_LSM: The expected net present value of the firm at time t=0.
    '''
    override_params = override_params or {}

    # Extract base parameters and apply overrides
    def param(name, fallback):
        return override_params.get(name, fallback)

    firm_name = p.get_name(gvkey)  # Firm name

    R_0 = p.get_R_0(gvkey)  # Initial revenue in millions per quarter
    L_0 = p.get_L_0(gvkey)  # Initial Loss-carryforward in millions per quarter
    X_0 = p.get_X_0(gvkey)  # Initial cash balance in millions per quarter
    CapEx_Ratio_0 = p.get_Initial_CapEx_Ratio(gvkey)  # Initial CapEx ratio to revenue
    CapEx_Ratio_longterm = p.get_Long_term_CAPEX(gvkey)  # Long-term CapEx ratio to revenue
    Dep_Ratio = p.get_Depreciation_Ratio(gvkey)  # Depreciation ratio to PPE
    PPE_0 = p.get_PPE_0(gvkey) # Initial PPE (Property, Plant, Equipment) in millions

    mu_0 = param("mu_0", p.get_mu_0())
    sigma_0 = param("sigma_0", p.get_sigma_0())
    eta_0 = param("eta_0", p.get_eta_0(gvkey))

    mu_mean = param("mu_mean", p.get_mu_mean())
    sigma_mean = param("sigma_mean", p.get_sigma_mean())
    
    taxrate = p.get_taxrate(gvkey)  # Corporate tax rate
    r_f = param("r_f", p.get_r_f())

    kappa_mu = param("kappa_mu", 0.09)
    kappa_sigma = param("kappa_sigma", p.get_kappa_sigma())
    kappa_eta = param("kappa_eta", p.get_kappa_eta())
    kappa_gamma = param("kappa_gamma", p.get_kappa_gamma())
    kappa_phi = param("kappa_phi", p.get_kappa_phi())
    kappa_capex = param("kappa_capex", p.get_kappa_capex())

    gamma_0 = p.get_gamma_0(gvkey)  # Initial cost ratio to revenue
    gamma_mean = p.get_gamma_mean(gvkey)  # Mean-reversion level for cost ratio to revenue
    phi_0 = param("phi_0", p.get_phi_0())
    phi_mean = param("phi_mean", p.get_phi_mean())


    lambda_R = param("lambda_R", p.get_lambda_R())
    lambda_mu = param("lambda_mu", p.get_lambda_mu())
    lambda_gamma = param("lambda_gamma", p.get_lambda_gamma())


    T = p.get_T()  # Time horizon in years
    dt = p.get_dt() # Time step
    M = param("M", p.get_exit_multiple()) # Exit multiple
    simulations = p.get_simulations()  # Number of Monte Carlo runs
    seasonal_factors = p.get_seasonal_factors(gvkey)  # Seasonal factors for revenue

    financing_cost = param("financing_cost", p.get_financing_cost())
    financing_grid = param("financing_grid", p.get_financing_grid(gvkey))
    C_max = p.get_C_max(gvkey)  # Maximum cash balance for financing decision

    num_steps = p.get_num_steps()

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
    financing_matrix = np.zeros(shape)  # Track financing amounts per time step
    beta_matrix = np.zeros((num_steps, 4))  # Store 4 regression coefficients for each timestep


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
    mu[:, 0] = mu_0 # Initial growth rate 
    gamma[:, 0] = gamma_0  # Initial cost ratio to revenue
    phi[0] = phi_0  # Initial volatility of cost ratio to revenue
    sigma[0] = sigma_0  # Initial revenue volatility
    eta[0] = eta_0  # Initial expected growth rate volatility
    L[:, 0] = L_0 # Initial loss carry-forward
    bankruptcy[:, 0] = False # Initial bankruptcy indicator
    

    # ------------------------------------------------------------
    # Forward Monte‑Carlo simulation
    # ------------------------------------------------------------
    for t in range(1, num_steps):

        # Get current quarter for seasonality adjustment
        quarter = (t % 4) if (t % 4) != 0 else 4
        seasonal_factor = seasonal_factors.get(quarter)
        
        # Update revenue using stochastic process and seasonal factor
        R[:, t] = R[:, t-1] * np.exp(
                (mu[:, t-1] - lambda_R * sigma[t-1] - 0.5 * sigma[t-1]**2) * dt + sigma[t-1] * np.sqrt(dt) * Z_R[:, t] # eq28, SchosserStröbele
            ) * seasonal_factor
        
        # Update growth rate with mean-reversion
        mu[:, t] = np.exp(-kappa_mu * dt) * mu[:, t-1] + (1 - np.exp(-kappa_mu * dt)) * (mu_mean - ((lambda_mu*eta[t-1])/(kappa_mu))) + np.sqrt((1 - np.exp(-2*kappa_mu*dt))/(2*kappa_mu)) * eta[t-1] * Z_mu[:, t] # eq29 i SchosserStröbele

        # Gamma (cost ratio to revenue)
        gamma[:, t] = np.exp(-kappa_gamma*dt) * gamma[:, t-1] + (1 - np.exp(-kappa_gamma*dt)) * (gamma_mean - ((lambda_gamma * phi[t-1])/(kappa_gamma))) + np.sqrt((1 - np.exp(-2*kappa_gamma*dt))/(2*kappa_gamma)) * phi[t-1] * Z_gamma[:, t] 


        # Sigma (volatility in revenue)
        sigma[t] = sigma[0] * np.exp(-kappa_sigma * t) + sigma_mean * (1 - np.exp(-kappa_sigma * t)) # eq19 SM2000 og eq32 SchosserStröbele
        
        # Update expected growth rate volatility
        eta[t] = eta[0] * np.exp(-kappa_eta * t) # eq20, SM2000 og schosser strobele eq33

        # Phi
        phi[t] = np.exp(-kappa_phi * t) * phi[0] + (1 - np.exp(-kappa_phi * t)) * phi_mean # eq34 in SchosserStrobele and eq30 in SM2001

        # CapEx ratio
        CapEx_ratio[t] = CapEx_ratio[0] * np.exp(-kappa_capex * t) + CapEx_Ratio_longterm * (1 - np.exp(-kappa_capex * t)) # Mean reverting Capex ratio.

        # Cost 
        Cost[:, t] = gamma[:, t] * R[:, t] # We use total cost ratio, gamma is excluding depreciation og amortization, but including interest expense.

        # Depreciation
        Dep[:, t] = Dep_Ratio * PPE[:, t-1]

        # Update CapEx
        CapEx[:, t] = CapEx_ratio[t] * R[:, t]

        # PPE
        PPE[:, t] = PPE[:, t-1] - Dep[:, t] + CapEx[:, t] 

        # Compute Tax in absolute value (eq14 in SchosserStrobele)
        taxable_income = R[:, t] - Cost[:, t] - Dep[:, t] - L[:, t-1]
        Tax[:, t] = np.where(taxable_income <= 0, 0, taxable_income * taxrate)

        # Update Net Operating Profit After Tax (NOPAT)
        NOPAT[:, t] = R[:, t] - Cost[:, t] - Dep[:, t] - Tax[:, t]

        # Compute Loss Carryforward
        used_loss = NOPAT[:, t] + Tax[:, t]
        L[:, t] = np.where(L[:, t-1] > used_loss, L[:, t-1] - used_loss, 0)

        # Update cash balance
        X[:, t] = X[:, t-1] + (r_f * X[:, t-1] + NOPAT[:, t] + Dep[:, t] - CapEx[:, t]) * dt

    # ------------------------------------------------------------
    # Longstaff‑Schwartz backward sweep with financing option
    # ------------------------------------------------------------
    discount = np.exp(-r_f*dt)  # Discount factor for cash flows

    NAV_proxy = PPE[:,-1]

    terminal = X[:, -1] + M * NAV_proxy

    V            = np.zeros_like(X)  # Value function
    V[:,-1]      = terminal  # Terminal value at maturity
    
    abandonment_value = 0.0  # Maybe later include a salvage value here, following Gamba
    bankrupt_now = np.zeros(simulations, dtype=bool)  # Track bankruptcy at time t
    bankruptcy = np.zeros((simulations, num_steps), dtype=bool)  # Track bankruptcy for all time steps

    r_squared = np.full(num_steps, np.nan)
    adj_r_squared = np.full(num_steps, np.nan)
    rmse = np.full(num_steps, np.nan)

    for t in range(num_steps - 2, -1, -1):  # Backward iteration over time steps
        cont_disc = discount * V[:, t + 1]  # Discounted continuation value

        # Identify paths that are not bankrupt and below the cash cutoff
        # The following lines are used to determine the paths that will be used for regression
        # Right now, the 20% firms with lowest cash, that are not bankrupt, are used for regression.
        percentile_cutoff = 20  # Percentile cutoff for cash balance
        not_bankrupt = ~bankruptcy[:, t+1]  # Paths that are not bankrupt
        nonbankrupt_cash = X[:, t][not_bankrupt]  # Cash balance of non-bankrupt paths
        if nonbankrupt_cash.size == 0:
            # Fallback: use all firms in the simulation
            valid_regression_paths = np.zeros(simulations, dtype=bool)
        else:
            cash_cutoff = np.percentile(nonbankrupt_cash, percentile_cutoff)
            valid_regression_paths = (X[:, t] <= cash_cutoff) & not_bankrupt

        min_paths = 10  # Minimum number of paths for regression
        if np.sum(valid_regression_paths) >= min_paths:
            B = basis(X[valid_regression_paths, t], R[valid_regression_paths, t])  # Basis for regression
            Y = cont_disc[valid_regression_paths]  # Discounted continuation value for valid paths
        else:  # If not enough paths, use all paths for regression
            B = basis(X[:, t], R[:, t])
            Y = cont_disc

        beta, *_ = np.linalg.lstsq(B, Y, rcond=None)  # Fit regression model (beta_0 + beta_1*X + beta_2*R + beta_3*X^2)
        beta_matrix[t, :] = beta  # Store regression coefficients
        
        C_hat_0 = basis(X[:, t], R[:, t]) @ beta  # Predicted continuation value given current cash and revenue
        
        # Compute regression diagnostics
        Y_pred = B @ beta
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        n = len(Y)
        p1 = B.shape[1]

        # R²
        r_squared[t] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Adjusted R²
        if ss_tot > 0 and n > p1:
            adj_r_squared[t] = 1 - (1 - r_squared[t]) * (n - 1) / (n - p1 - 1)

        # RMSE
        rmse[t] = np.sqrt(np.mean((Y - Y_pred) ** 2))


        best_val = C_hat_0.copy()  # Initialize best value with predicted continuation value
        best_f = np.zeros(simulations)  # Initialize best financing choice

        for f in financing_grid[1:]:  # iterate over financing amounts (excluding 0)
            X_tmp = X[:, t] + f  # Cash balance after financing
            C_tmp = basis(X_tmp, R[:, t]) @ beta  # Predicted continuation value after financing
            val = -financing_cost * f + C_tmp  # Value after financing
            can_finance = (X[:, t] < C_max) & (val > 0)  # Paths that can finance and have positive value
            mask = (val > best_val) & can_finance # Identify paths where financing is better than current value
            best_val[mask] = val[mask]  # Update best value
            best_f[mask] = f  # Update financing choice
        
        if t == 0:
            best_f[:] = 0
            best_val[:] = C_hat_0  # Set financing choice to 0 for t=0

        # Identify bankrupt paths: cash < 0 and no financing chosen or value < 0
        bankrupt_now = ((X[:, t] < 0) & (best_f == 0)) | (best_val < 0)  # Paths that are bankrupt now

        # Record bankruptcies for analysis
        bankruptcy[bankrupt_now, t:] = True

        # Set value to abandonment value for bankrupt paths
        best_val[bankrupt_now] = abandonment_value

        financing_matrix[:, t] = best_f  # Store chosen financing amount at time t
        financing_matrix[bankrupt_now, t:] = 0.0

        # Adjust cash flow for financing, but not for bankrupt paths
        cash_flow = -financing_cost * best_f * (~bankrupt_now)
        V[:, t] = cash_flow + best_val
        V[:, t] = np.maximum(V[:, t], 0.0)

    total_discounted_financing = np.sum(financing_matrix * np.exp(-r_f * np.arange(num_steps) * dt), axis=1)

    V0_LSM = np.mean(V[:,0]- total_discounted_financing) - X_0  # Hadde i prinsippet ikke trengt å ta snittet for alle burde være like når t=0. pga lik cash og revenue.
    
    num_bankruptcies = np.sum(np.any(bankruptcy, axis=1)) # Count bankruptcies across all time steps

    # ------------------------------------------------------------
    # Save simulation results to file
    # ------------------------------------------------------------
    if save_to_file:
        # Create a timestamped filename
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
                "rho_R_mu": 0,
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
                "financing_cost": financing_cost,
                "financing_grid": financing_grid,
                "seasonal_factors": seasonal_factors,
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
                "terminal_value": terminal,
                "V0": V0_LSM,
                "num_bankruptcies": num_bankruptcies,
                "financing": financing_matrix,
                "betas": beta_matrix,
            }
        }

        ### Commented out add to all sims, only keep latest sim. ###

        # filename_complete = f"{gvkey}_sim_results_{timestamp}.pkl"
        filename_latest_sim = f"v8_{gvkey}_latest_sim_results.pkl"

        # Save to disk
        # output_dir_all = "simulation_outputs_all"
        output_dir_latest = "simulation_outputs_latest"
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
    return V0_LSM, num_bankruptcies



if __name__ == "__main__":
    # Simulate firm value for each company in the list
    for gvkey in p.COMPANY_LIST:
        simulate_firm_value(gvkey, save_to_file=True)

    # gvkey = 232646
    # simulate_firm_value(gvkey, save_to_file=True)


