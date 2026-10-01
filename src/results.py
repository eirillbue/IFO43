# ------------------------------------------------------------
# results.py
# ------------------------------------------------------------
# This script is used to present the results from a saved model.
# The scripts contains different methods for plotting and printing 
# results.
# ------------------------------------------------------------
# Authors: 
# Per Inge Notland
# David Taraldsen
# ---
# Date: 05.06.2025
# ------------------------------------------------------------

import pickle
import parameters as p
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_latest_simulation_results(gvkey, version=8):
    """
    Load simulation results from a pickle file.

    Parameters:
    - gvkey (str): The gvkey of the firm.

    Returns:
    - dict: A dictionary containing the simulation results.
    """
    # Path to the file you saved earlier
    filepath = f"simulation_outputs_latest/v{version}_{gvkey}_latest_sim_results.pkl"
    # filepath = f"models/v6_{gvkey}_latest_sim_results.pkl"

    # Load the file
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # Now `data` is a dictionary with two top-level keys: "parameters" and "results"
    return data

def get_model(gvkey, version=8):
    """
    Load simulation results from a pickle file.

    Parameters:
    - gvkey (str): The gvkey of the firm.

    Returns:
    - dict: A dictionary containing the simulation results.
    """
    # Path to the file you saved earlier
    filepath = f"models/v{version}_{gvkey}_latest_sim_results.pkl"

    # Load the file
    print("Loading model from:", filepath)
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # Now `data` is a dictionary with two top-level keys: "parameters" and "results"
    return data


def print_main_results(firm_data):
    '''
    Print the results of the simulation.
    '''
    print("\nMonte Carlo Simulation Results")
    print("===================================")
    print("GVKEY:", firm_data["gvkey"])
    print("Name:", firm_data["name"])
    print("Timestamp: ", firm_data["timestamp"])
    V0 = firm_data["results"]["V0"]
    print("\nEstimated Value (V0): ", V0.round(2), "millions")
    num_bankruptcies = firm_data["results"]["num_bankruptcies"]
    simulations = firm_data["parameters"]["simulations"]
    print("Bankrupt simulations:", num_bankruptcies, "out of", simulations, ",   ", (num_bankruptcies/simulations)*100, "%")


def print_all_parameters(firm_data):
    '''
    Print all parameters.
    '''
    print("\nModel Parameters")
    print("===================================")
    for key, value in firm_data["parameters"].items():
        print(f"{key}: {value}")


def print_revenue_distributions(data):
    R = data["results"]["R"]
    # Compute quantiles
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95, 0.995]
    revenue_quantiles = np.quantile(R, quantiles, axis=0)

    # Compute quantiles
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.995]
    revenue_quantiles = np.quantile(R[:, [3, 11, 19, 27, 39, 59, 79, 99]], quantiles, axis=0)  # Extract values for 1, 3, 5, 7, 10 years forward

    # Create a DataFrame
    years_forward = [1, 3, 5, 7, 10, 15, 20, 25]
    percentile_labels = ["5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%", "99.5%", "Mean"]
    revenue_table = pd.DataFrame(revenue_quantiles, columns=years_forward, index=percentile_labels[:-1])
    revenue_table.loc["Mean"] = np.mean(R[:, [3, 11, 19, 27, 39, 59, 79, 99]], axis=0)

    # Print the table
    print("\nTable 4. Revenue Distributions (millions)\n")
    print(revenue_table)


def plot_firm_value_distribution(data, curr):
    '''
    Plot the distribution of firm values from the simulation,
    showing the middle 90% of values (5th to 95th percentile).
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    # Parameters and terminal values
    # r_f = data["parameters"]["r_f"]
    r_f = 0.03055489
    T = data["parameters"]["T"]
    terminal_value = data["results"]["terminal_value"]
    cash = data["results"]["X"][:, -1]  # Cash at terminal time
    name = data["name"]

    # Discounted firm value
    firm_value = (terminal_value + cash) * np.exp(-r_f * T)

    # Middle 90% bounds
    lower_pct, upper_pct = 5, 95
    lower_bound = np.percentile(firm_value, lower_pct)
    upper_bound = np.percentile(firm_value, upper_pct)
    mid_values = firm_value[(firm_value >= lower_bound) & (firm_value <= upper_bound)]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(mid_values, bins=100, density=True, alpha=0.7,
             color='skyblue', edgecolor='grey', label=f'Middle {upper_pct - lower_pct}%')

    # Mean and median
    mean_val = np.mean(firm_value)
    median_val = np.median(firm_value)

    plt.axvline(mean_val, color='blue', linestyle='--', linewidth=2,
                label=f'Mean (All): {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2,
                label=f'Median (All): {median_val:.1f}')

    plt.title(f"Distribution of Discounted Firm Values before Financing option (Middle 90%), {name}")
    plt.xlabel(f"Present Value (Millions {curr})")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_combined_bankruptcy_timeline(data):
    import matplotlib.pyplot as plt
    import numpy as np

    bankruptcy = data["results"]["bankruptcy"]  # shape: (simulations, num_steps)
    name = data["name"]
    n_sim = bankruptcy.shape[0]

    # New bankruptcies (transitions from 0 to 1)
    new_bankruptcies = np.diff(bankruptcy.astype(int), axis=1)
    new_bankruptcies = np.maximum(new_bankruptcies, 0)
    bankruptcies_per_step = np.sum(new_bankruptcies, axis=0)

    # Time axis (note: np.diff reduces length by 1)
    time = np.arange(1, bankruptcy.shape[1])

    # Cumulative bankruptcy rate (percentage)
    cumulative_rate = np.mean(bankruptcy, axis=0) * 100

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Y-axis: New bankruptcies
    ax1.bar(time, bankruptcies_per_step, color='salmon', alpha=0.8, label="New Bankruptcies")
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel(f"New Bankruptcies", color='salmon')
    ax1.tick_params(axis='y', labelcolor='salmon')

    # Right Y-axis: Cumulative bankruptcy rate
    ax2 = ax1.twinx()
    ax2.plot(np.arange(bankruptcy.shape[1]), cumulative_rate, color='blue', linewidth=2, label="Cumulative Bankruptcy Rate (%)")
    ax2.set_ylabel("Bankrupt Firms (%)", color='blue')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='blue')

    # Titles and grid
    plt.title(f"Bankruptcy Timeline, {name}")
    fig.tight_layout()
    ax1.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)

    # Optional: Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()



def plot_financing_percent(data, curr):
    """
    Plot the amount of financing raised over time, and percentage of firms receiving financing.

    Parameters:
    - data: Dictionary loaded from the simulation results (via get_latest_simulation_results)
    - curr: Currency label to display in y-axis label
    """
    import matplotlib.pyplot as plt
    import numpy as np

    financing = data["results"].get("financing")
    bankruptcy = data["results"].get("bankruptcy")

    # Create a copy of the financing matrix where bankrupt firms are excluded
    financing_matrix_clean = financing.copy()

    # Mask financing where firm is bankrupt (1 in bankruptcy matrix)
    financing_matrix_clean[bankruptcy == True] = 0

    if financing is None:
        print("No financing data found in results.")
        return
    
    name = data["name"]
    num_simulations = financing_matrix_clean.shape[0]
    num_steps = financing_matrix_clean.shape[1]
    time = np.arange(num_steps)

    max_financing = data["parameters"]["financing_grid"][-1]
    

    total_raised = np.sum(financing_matrix_clean, axis=0)
    percent_financed = (np.sum(financing_matrix_clean > 0, axis=0) / num_simulations) * 100

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left axis: total financing raised
    ax1.bar(time, total_raised, color='dodgerblue', alpha=0.7, label=f'Total Financing Raised (Millions {curr})')
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel(f"Financing Raised (Millions {curr})", color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.grid(True)
    ax1.set_ylim(top=max_financing*1000000)

    # Right axis: % of firms that raised financing
    ax2 = ax1.twinx()
    ax2.plot(time, percent_financed, color='darkred', linestyle='--', linewidth=2, label='Firms Receiving Financing (%)')
    ax2.set_ylabel("Firms Receiving Financing (%)", color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(0, 100)

    plt.title(f"Financing, {name}")
    fig.tight_layout()

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.show()



def plot_revenue(data, curr):

    R = data["results"]["R"]
    name = data["name"]
    t = np.arange(R.shape[1])

    mean = np.mean(R, axis=0)
    p10 = np.percentile(R, 10, axis=0)
    p90 = np.percentile(R, 90, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(t, mean, color='blue', linewidth=2, label="Mean Revenue")
    plt.fill_between(t, p10, p90, color='blue', alpha=0.2, label="10th–90th Percentile")

    plt.xlabel("Quarters")
    plt.ylabel(f"Revenue (Millions {curr})")
    plt.title(f"Revenue, {name}")
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_cash(data, curr):
    X = data["results"]["X"]  # Cash matrix (simulations x timesteps)
    name = data["name"]
    t = np.arange(X.shape[1])

    mean = np.mean(X, axis=0)
    p10 = np.percentile(X, 10, axis=0)
    p90 = np.percentile(X, 90, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(t, mean, color='green', linewidth=2, label="Mean Cash")
    plt.fill_between(t, p10, p90, color='green', alpha=0.2, label="10th–90th Percentile")

    plt.xlabel("Quarters")
    plt.ylabel(f"Cash (Millions {curr})")
    plt.title(f"Cash Balance, {name}")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    idx = 0
    for company in p.COMPANY_LIST:
        data = get_latest_simulation_results(company, version=8)
        curr = p.COMPANY_CURRENCIES[idx]
        idx += 1
        print_main_results(data)
        # print_all_parameters(data)
        # plot_revenue(data, curr=curr)
        # plot_firm_value_distribution(data, curr=curr)
        # plot_cash(data, curr=curr)
        # plot_financing_percent(data, curr=curr)
        # plot_combined_bankruptcy_timeline(data)

    # idx = 0
    # for company in p.COMPANY_LIST:
    #     data = get_model(company, version=8)
    #     curr = p.COMPANY_CURRENCIES[idx]
    #     idx += 1
    #     print_main_results(data)
    #     print_all_parameters(data)
    #     plot_revenue(data, curr=curr)
    #     plot_firm_value_distribution(data, curr=curr)
    #     plot_cash(data, curr=curr)
    #     plot_financing_percent(data, curr=curr)
    #     plot_combined_bankruptcy_timeline(data)

