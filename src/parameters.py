# ------------------------------------------------------------
# parameters.py
# ------------------------------------------------------------
# This script contains the parameters used in 
# the valuation model NT_2025.
# ---
# Authors: 
# Per Inge Notland
# David Taraldsen
# ---
# Date: 05.06.2025
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FUNDAMENTALS_CSV_PATH = 'data/all_companies_quarterly.csv'
FUNDAMENTALS_Y2D_CSV_PATH = 'data/all_companies_y2d.csv'
STOCK_PRICES_CSV_PATH = 'data/stock_prices.csv'
STOXX_600_CSV_PATH = 'data/stoxx600_monthly.csv'

COMPANY_LIST = [103342, 225094, 225597, 232646, 245628, 318456, 328809, 329260]
COMPANY_NAMES = ['SSE', 'VESTAS', 'FORTUM', 'ORSTED', 'NORDEX', 'SCATEC', 'NEOEN', 'ENCAVIS']

COMPANY_CURRENCIES = ['GBP', 'EUR', 'EUR', 'DKK', 'EUR', 'NOK', 'EUR', 'EUR']
COMPANY_EV_0= [27780, 13460, 13940, 180850, 2940, 35860, 10900, 5120]
EURNOK = 11.79 
EURGBP = 0.83 
EURDKK = 7.46 # 1.januar tall på alle
COMPANY_CURRENCY_FACTORS = [EURGBP, 1, 1, EURDKK, 1, EURNOK, 1, 1]

### LOAD DATA ###
try:
    FUNDAMENTALS = pd.read_csv(FUNDAMENTALS_CSV_PATH)
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {FUNDAMENTALS_CSV_PATH}: {e}")

try:
    FUNDAMENTALS_Y2D = pd.read_csv(FUNDAMENTALS_Y2D_CSV_PATH)
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {FUNDAMENTALS_Y2D_CSV_PATH}: {e}")

try:
    STOCK_PRICES = pd.read_csv(STOCK_PRICES_CSV_PATH, low_memory=False)
    STOCK_PRICES = STOCK_PRICES[STOCK_PRICES['iid'] == '01W']   # Filter for '01W' iid
except Exception as e:
    raise FileNotFoundError(f"Error reading stock prices CSV file at {STOCK_PRICES_CSV_PATH}: {e}")

try:
    STOXX_600 = pd.read_csv(STOXX_600_CSV_PATH)
    STOXX_600['datadate'] = pd.to_datetime(STOXX_600['datadate'], errors='coerce')
    STOXX_600 = STOXX_600.sort_values('datadate')
    prices = STOXX_600.set_index('datadate')['prccm']

    # Resample to quarterly prices (using last price of each quarter)
    quarterly_prices = prices.resample("QE").last()

    # Calculate log returns
    STOXX_QUARTERLY = np.log(quarterly_prices / quarterly_prices.shift(1)).dropna()

    # Convert to DataFrame and add 'fyearq' and 'fqtr'
    STOXX_QUARTERLY = STOXX_QUARTERLY.to_frame('log_return_market')
    STOXX_QUARTERLY['fyearq'] = STOXX_QUARTERLY.index.year
    STOXX_QUARTERLY['fqtr'] = STOXX_QUARTERLY.index.quarter

    # Calculate standard deviation of returns
    SIGMA_MARKET = STOXX_QUARTERLY['log_return_market'].std()
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {STOXX_600_CSV_PATH}: {e}")


### CLEAN UP DATA ###
# FORTUM gjorde oppkjøp og solgte uniper, så vi må fjerne det fra dataene.
# Subtract Uniper (in Fortum) sales from Fortum sales. From company report/excel sheet.
FUNDAMENTALS.loc[43, 'saleq'] = FUNDAMENTALS.loc[43, 'saleq'] - 11365
FUNDAMENTALS.loc[44, 'saleq'] = FUNDAMENTALS.loc[44, 'saleq'] - 13159
FUNDAMENTALS.loc[45, 'saleq'] = FUNDAMENTALS.loc[45, 'saleq'] - 19990
FUNDAMENTALS.loc[46, 'saleq'] = FUNDAMENTALS.loc[46, 'saleq'] - 19770
FUNDAMENTALS.loc[47, 'saleq'] = FUNDAMENTALS.loc[47, 'saleq'] - 15893

# Må endre cogs også. Uniper sales - uniper ebitda. 
FUNDAMENTALS.loc[43, 'cogsq'] = FUNDAMENTALS.loc[43, 'cogsq'] - (11365 - 184) # Justert for cogs, sga og xopro gjennom en ekstra utregning.
FUNDAMENTALS.loc[44, 'cogsq'] = FUNDAMENTALS.loc[44, 'cogsq'] - (13159 + 147)
FUNDAMENTALS.loc[45, 'cogsq'] = FUNDAMENTALS.loc[45, 'cogsq'] - (19990 - 819)
FUNDAMENTALS.loc[46, 'cogsq'] = FUNDAMENTALS.loc[46, 'cogsq'] - (19770 - 868)
FUNDAMENTALS.loc[47, 'cogsq'] = FUNDAMENTALS.loc[47, 'cogsq'] - (15893 + 17)


### PARAMETERS ###
def get_name(gvkey):
    """
    Returns the name of the company based on the gvkey.
    """
    try:
        index = COMPANY_LIST.index(gvkey)
        return COMPANY_NAMES[index]
    except ValueError:
        return None


def get_R_0(gvkey, df=FUNDAMENTALS):
    '''
    Initial Revenue
    '''
    return df[df['gvkey'] == gvkey]['saleq'].iloc[-1] # Last quarter revenue


def get_L_0(gvkey, df=FUNDAMENTALS):
    '''
    Initial Loss Carryforward.
    Loss carry-forward is received directly from company financial reports.
    '''
    # At first, we thought we could use deferred assets/liabilities
    # Concluded with using only the tax loss carried forward presented in company reports
    # where we could find it.

    def_tax_assets = {
        103342: 58.4, # SSE in MGBP (Q3 2024)
        225094: 722, # Vestas
        225597: 845, # Fortum
        232646: 9250, # Orsted i MDKK
        245628: 530.669, # Nordex i MEUR
        318456: 1551, # Scatec i MNOK
        328809: 116.2, # Neoen i MEUR
        329260: 11.058, # Encavis i MEUR (Q3 2024)
    }

    def_tax_liabilities = {
        103342: 1639.7, # SSE in MGBP (Q3 2024)
        225094: 179, # Vestas
        225597: 386, # Fortum
        232646: 2433, # Orsted i MDKK
        245628: 203.675, # Nordex i MEUR
        318456: 671, # Scatec i MNOK
        328809: 185.3, # Neoen i MEUR
        329260: 152.254, # Encavis i MEUR (Q3 2024)
    }

    loss_carryforward = {
        103342: 0, # SSE in MGBP (Q3 2024)
        225094: 707, # Vestas Annual 2024 M EUR
        225597: 907, # Fortum Annual 2024 MEUR
        232646: 4198, # Orsted i MDKK Annual 2024
        245628: 406.074, # Nordex i MEUR Annual 2024 (tax loss) + 0.190 interest + 36.907 tax credits
        318456: 4455, # Scatec i MNOK Annual 2024
        328809: 196.7, # Neoen i MEUR Annual 2023 (Tax loss carryforwards and unused tax credits)
        329260: 24015, # Encavis i MEUR (Annual 2023)
    }

    return loss_carryforward.get(gvkey)


def get_Initial_CapEx_Ratio(gvkey, df=FUNDAMENTALS_Y2D, n_quarters=8):
    '''
    Capex som ratio av revenue (capex/sales) - y2d siste 8 kvartaler
    '''
    ratios = []
    firm_data = df[df['gvkey'] == gvkey].sort_values(by='datadate')
    recent_data = firm_data.tail(n_quarters)
    for _, row in recent_data.iterrows():
        capex = row.get('capxy', 0)
        revenue = row.get('saley', 0)
        if revenue == 0:
            continue
        ratios.append(capex / revenue)

    return np.nanmean(ratios) if ratios else None


def get_Long_term_CAPEX(gvkey, df=FUNDAMENTALS_Y2D, n_quarters=8):
    '''
    Dep_ratio * (initial ppe / initial revenue)
    '''
    ppe = get_PPE_0(gvkey)
    revenue = get_R_0(gvkey)
    dep = get_Depreciation_Ratio(gvkey)
    return dep * (ppe / revenue)


def get_Depreciation_Ratio(gvkey, df=FUNDAMENTALS, n_quarters=8):
    '''
    Returns the average depreciation ratio (dpq/ppentq) for the last n_quarters of a firm.
    '''
    ratios = []  # list to store ratios
    firm_data = df[df['gvkey'] == gvkey].sort_values(by='datadate')  # sort by date
    recent_data = firm_data.tail(n_quarters)  # get the last n_quarters of data
    for _, row in recent_data.iterrows():  # iterate over the rows
        dep = row.get('dpq', 0)  # get depreciation
        ppe = row.get('ppentq', 0)  # get property, plant, and equipment
        if ppe == 0:  # avoid division by zero
            continue  
        ratios.append(dep / ppe)  # calculate ratio and append to list

    return np.mean(ratios) if ratios else None  # return mean of ratios if not empty


def get_PPE_0(gvkey, df=FUNDAMENTALS):
    '''
    Property, Plant and Equipment (ppentq) - siste kvartal
    '''
    return df[df['gvkey'] == gvkey]['ppentq'].iloc[-1]

def get_exit_multiple():
    """
    Exit multiple (M) is Multiple on PPE as a proxy for NAV - used in valuation
    """
    return 1

def get_X_0(gvkey, df=FUNDAMENTALS):
    '''
    Initial Cash Balance
    '''
    return df[df['gvkey'] == gvkey]['cheq'].iloc[-1]


def get_mu_0(df=FUNDAMENTALS):
    """
    Initial expected rate of growth in revenues (mu_0)
    """
    # Get the most recent growth rate
    growth_rates = []
    for company in COMPANY_LIST:
        q12024 = df[(df['gvkey'] == company) & (df['fqtr'] == 1) & (df['fyearq'] == 2024)]['saleq'].values[0]
        q12020 = df[(df['gvkey'] == company) & (df['fqtr'] == 1) & (df['fyearq'] == 2020)]['saleq'].values[0]
        cagr = ((q12024 / q12020) ** (1/16)) - 1
        growth_rates.append(cagr)
        # print(f"Company {get_name(company)}: CAGR from Q1 2020 to Q1 2024 is {cagr:.4f}")
    return np.mean(growth_rates)


def get_sigma_0(df=FUNDAMENTALS, seasonal=True):
    """
    Initial volatility of revenues (sigma_0) - Mean of all company quarterly standard deviations.
    Adjusts for seasonality in revenue by normalizing saleq using seasonal factors.
    """

    company_volatilities = []

    for company in COMPANY_LIST:
        seasonal_factors = get_seasonal_factors(company, df, seasonal=seasonal)

        # Extract quarterly revenue data for each company
        revenues = df[df['gvkey'] == company].sort_values(by=['fyearq', 'fqtr'])[['fyearq', 'fqtr', 'saleq']].copy()

        # Drop rows with NaN in 'saleq'
        revenues = revenues.dropna(subset=['saleq'])

        # Adjust 'saleq' for seasonality
        revenues['adjusted_saleq'] = revenues.apply(lambda row: row['saleq'] / seasonal_factors.get(row['fqtr'], 1.0), axis=1)
        # Calculate quarterly revenue growth rates on adjusted revenues
        revenues['growth_rate'] = revenues['adjusted_saleq'].pct_change()

        # Drop NaN values (first row will have NaN)
        growth_rates = revenues['growth_rate'].dropna().values

        # Compute standard deviation only if there are enough data points
        if len(growth_rates) > 1:
            company_volatilities.append(np.std(growth_rates, ddof=1))  # Sample std dev

    # Compute the average volatility across companies (ignore NaN values)
    return np.nanmean(company_volatilities) if company_volatilities else np.nan

def get_eta_0(gvkey, default=True, frequency='quarterly', df=STOCK_PRICES):
    """
    Estimates eta_0, the initial volatility of expected growth rate in revenues.
    Uses historical stock price log-returns to infer volatility.
    Vi bruker iid=01W, som er hvilken issue av aksjen som er i bruk.
    En del antagelser vi må gjøre her, men tror dette er beste måten.
    """
    firm_data = df[df['gvkey'] == gvkey].dropna(subset=['prccd'])
    firm_data['datadate'] = pd.to_datetime(df['datadate'], errors='coerce')
    firm_data = firm_data.sort_values('datadate')


    if firm_data.empty:
        raise ValueError(f"No price data found for gvkey {gvkey}")

    prices = firm_data['prccd'].values

    if len(prices) < 2:
        raise ValueError(f"Not enough price data to compute returns for gvkey {gvkey}")

    # Compute log returns
    log_returns = np.diff(np.log(prices))

    # Calculate sample standard deviation of daily returns
    volatility = np.std(log_returns, ddof=1)

    # Scale to desired frequency
    if default == True:  # 
        return 0.03
    elif frequency == 'daily':
        return volatility
    elif frequency == 'monthly':
        return volatility * np.sqrt(21)   # ~21 trading days per month
    elif frequency == 'quarterly':
        return volatility * np.sqrt(63)   # ~63 trading days per quarter
    elif frequency == 'annual':
        return volatility * np.sqrt(252)
    else:
        raise ValueError("Frequency must be 'daily', 'monthly', 'quarterly', or 'annual'.")


def get_rho_R_mu():
    """
    Correlation between revenue and growth rate
    SchwartzMoon antar 0.0
    "Estimated from past company or cross-sectional data"

    Consider the situation in which the firm is in a competitive environment such that 
    increases in profit margins (i.e. decreases in variable costs) are associated with 
    decreases in growth rates in revenues. This implies a positive correlation between
    variable costs and growth rates in revenues. We run the program with the same
    data described above, but assuming a correlation of 0.5. The market price of risk
    and the volatility of the growth rate had to be adjusted slightly to match the volatility
    and the beta of the stock. The model price decreased only slightly to $21.89.
    The effect of this correlation on prices increases, however, with the volatility of
    variable costs.
    """
    return 0.0 # Kan la denne være 0


def get_mu_mean():
    """
    Mean-reversion level for growth rate (mu_mean)
    
    Schwartz Moon (2000):
    Rate of growth in revenues for a stable company in the same industry as the company being valued.
    """
    return 0.0075 # 3% annually.


def get_sigma_mean():    
    """
    Mean-reversion level for volatility (sigma_mean)
    """
    return 0.05


def get_taxrate(gvkey, df=FUNDAMENTALS):
    """
    Corporate tax rate (taxrate)
    """
    firm_data = df[df['gvkey'] == gvkey]

    if firm_data.empty:
        return None  # or np.nan
    
    try:
        taxes = firm_data['txtq'].iloc[-1]
        pretax_income = firm_data['piq'].iloc[-1]
        if pretax_income <= 0:
            return 0.24  # Default tax rate if pretax income is zero or negative
        taxrate = taxes / pretax_income
        if taxrate <= 0.15 or taxrate > 0.40:  # If the calculated tax rate is too low or too high, use a default value
            taxrate = 0.24  
        return taxrate
    except (IndexError, KeyError, ZeroDivisionError):
        return None

def get_r_f(freq='q'):
    """
    Risk-free rate (r_f)
    We have chosen a 25-year rate (to 2050) due to the net zero target and valuation horizon.
    """
    r_y = 0.03055489  # European 25Y AAA bond yield as of 2024-01-01 (3.055489%)
    
    if freq == 'y':
        return r_y
    elif freq == 'q':
        return (1 + r_y) ** (1/4) - 1


def get_kappa_mu(convergence=0.95):
    """
    Mean-reversion speed for expected growth rate (kappa_mu)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa


def get_kappa_sigma(convergence=0.95):
    """
    Mean-reversion speed for volatility (kappa_sigma)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa


def get_kappa_eta(convergence=0.95):
    """
    Mean-reversion speed for expected growth rate volatility (kappa_eta)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa

def get_kappa_capex(convergence=0.95):
    '''
    Mean reversion speed for capex
    '''
    kappa = -1 * np.log(1-convergence) / 100
    return kappa


def get_gamma_0(gvkey, df=FUNDAMENTALS, n_quarters=8):
    """
    Calculate the ratio of total costs to revenues for the last quarters of the firm.
    Without Depreciation and Interest. (EBITDA margin) Need to include Dep in montecarlo simulation.
    Denne er firmspesifikk, og er ikke snittet av alle selskapene i COMPANY_LIST.
    """
    firm_data = df[df['gvkey'] == gvkey].sort_values(by='datadate')

    if firm_data.empty:
        return None

    # Get the last `n_quarters` of data
    recent_data = firm_data.tail(n_quarters)

    ratios = []
    for _, row in recent_data.iterrows():
        revenue = row.get('saleq', None)
        cogs = row.get('cogsq', 0)
        cogs = 0 if pd.isna(cogs) else cogs
        sga = row.get('xsgaq', 0)
        sga = 0 if pd.isna(sga) else sga
        other_operating_expense = row.get('xoproq', 0)
        other_operating_expense = 0 if pd.isna(other_operating_expense) else other_operating_expense
        # depr = row.get('dpq', 0) or 0
        interest = row.get('xintq', 0) or 0

        if pd.isna(revenue) or revenue == 0:
            continue  # skip invalid or missing quarters

        operating_costs = cogs + sga + other_operating_expense # + depr + interest
        ratio = operating_costs / revenue
        ratios.append(ratio)

    if not ratios:
        return None

    return sum(ratios) / len(ratios)


def get_gamma_mean(gvkey):
    """
    Mean-reversion level for the ratio of total cost to revenues (gamma_mean)
    Mean of all initial company ratios. 
    Gazheli skriver noe om learning rate vi kan bruke her?
    """
    # Kommentert ut fordi den burde være firmspesifikk.
    # gammas = []
    # for company in COMPANY_LIST:
    #     gamma = get_gamma_0(company)
    #     if gamma is not None:
    #         gammas.append(gamma)

    return 0.9 * get_gamma_0(gvkey)


def get_kappa_gamma(convergence=0.95):
    """
    Mean-reversion speed for the ratio of total cost to revenues (kappa_gamma)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa


def get_phi_0(df=FUNDAMENTALS, n_quarters=8):
    """
    Initial volatility of the ratio of total cost to revenues (phi_0)
    Denne er ikke firmspesifikk, og er snittet av alle selskapene i COMPANY_LIST.
    """
    # Get the last `n_quarters` of data
    stds = []

    for company in COMPANY_LIST:
        firm_data = df[df['gvkey'] == company].sort_values(by='datadate')
        recent_data = firm_data.tail(n_quarters)
        ratios = []


        for _, row in recent_data.iterrows():
            revenue = row.get('saleq', None)
            cogs = row.get('cogsq', 0)
            cogs = 0 if pd.isna(cogs) else cogs
            sga = row.get('xsgaq', 0)
            sga = 0 if pd.isna(sga) else sga
            other_operating_expense = row.get('xoproq', 0)
            other_operating_expense = 0 if pd.isna(other_operating_expense) else other_operating_expense
            # depr = row.get('dpq', 0) or 0
            # interest = row.get('xintq', 0) or 0

            if pd.isna(revenue) or revenue == 0:
                continue  # skip invalid or missing quarters

            operating_costs = cogs + sga + other_operating_expense # + depr + interest
            ratio = operating_costs / revenue
            ratios.append(ratio)

        if not ratios:
            return None
        
        stds.append(np.std(ratios, ddof=1))  # Sample std dev
    
    return np.mean(stds) if stds else np.nan


def get_phi_mean():
    """
    Mean-reversion level for the volatility of the ratio of total cost to revenues (phi_mean)
    """
    return get_phi_0() / 2


def get_kappa_phi(convergence=0.95):
    """
    Mean-reversion speed for the volatility of the ratio of total cost to revenues (kappa_phi)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa

def get_sector_returns(df=FUNDAMENTALS):
    '''
    Converts sales to EUR using predefined currency factors,
    then calculates total sector revenue and log returns.
    '''
    # Pivot to get sales data in wide format
    df_pivoted = df.pivot(index=['fyearq', 'fqtr'], columns='gvkey', values='saleq')

    # Map gvkey to currency conversion factors
    gvkey_to_factor = dict(zip(COMPANY_LIST, COMPANY_CURRENCY_FACTORS))

    # Scale each company's sales to EUR
    for gvkey in df_pivoted.columns:
        if gvkey in gvkey_to_factor:
            df_pivoted[gvkey] = df_pivoted[gvkey] * gvkey_to_factor[gvkey]
        else:
            print(f"Warning: Missing currency factor for gvkey {gvkey}, skipping scaling.")

    # Calculate total revenue and log returns
    df_pivoted['total_revenue'] = df_pivoted.sum(axis=1)
    df_pivoted['Log_Returns_Revenue'] = np.log(df_pivoted['total_revenue'] / df_pivoted['total_revenue'].shift(1))
    df_pivoted['Log_Returns_Growth'] = df_pivoted['Log_Returns_Revenue'].diff()

    return df_pivoted



def get_lambda_R(df=FUNDAMENTALS, market_returns=STOXX_QUARTERLY, market_std=SIGMA_MARKET):
    """
    Calculates the market price of risk (lambda_R) for revenue.
    Computer correlation mellom log returns of revenue and log market returns.
    får ganske lik lambda som schwartz Moon, veldig lav nær 0.
    """
    revenue_data = get_sector_returns(df)
    market_data = market_returns.copy()

    # Merge revenue and market returns based on year and quarter
    merged_data = pd.merge(revenue_data, market_data, on=['fyearq', 'fqtr'], how='inner', suffixes=('_revenue', '_market'))

    # Compute correlation
    correlation = merged_data['Log_Returns_Revenue'].corr(merged_data['log_return_market']) # Kolonnene heter det i revenue og market
    # Compute lambda_R
    lambda_R = correlation * market_std

    return lambda_R


def get_lambda_mu(df=FUNDAMENTALS, market_returns=STOXX_QUARTERLY, market_std=SIGMA_MARKET):
    """
    Calculates the market price of risk (lambda_mu) for the expected rate of growth factor (mu).
    """

    # Assume we have a helper function that gives us mu values
    mu_data = get_sector_returns()

    # Copy market returns to ensure data integrity
    market_data = market_returns.copy()

    # Merge mu data and market returns based on year and quarter
    merged_data = pd.merge(mu_data, market_data, on=['fyearq', 'fqtr'], how='inner', suffixes=('_mu', '_market'))

    # Compute correlation between mu returns and market returns
    correlation = merged_data['Log_Returns_Growth'].corr(merged_data['log_return_market'])

    # Compute lambda_mu
    lambda_mu = correlation * market_std

    return lambda_mu


def get_lambda_gamma(df=FUNDAMENTALS, market_returns=STOXX_QUARTERLY, market_std=SIGMA_MARKET):
    """
    Calculates the market price of risk (lambda_gamma) for the ratio of total costs to revenues (gamma).
    """
    # Assume we have a helper function that gives us gamma values
    gamma_data = get_cost_ratio_table()

    # Copy market returns to ensure data integrity
    market_data = market_returns.copy()

    # Merge gamma data and market returns based on year and quarter
    merged_data = pd.merge(gamma_data, market_data, on=['fyearq', 'fqtr'], how='inner', suffixes=('_gamma', '_market'))

    # Compute correlation between gamma returns and market returns
    correlation = merged_data['Log_Returns_Gamma'].corr(merged_data['log_return_market'])

    # Compute lambda_gamma
    lambda_gamma = correlation * market_std

    return lambda_gamma


def get_cost_ratio_table(df=FUNDAMENTALS, company_list=COMPANY_LIST, n_quarters=None):
    """
    Returns a DataFrame of cost ratios for each company and each quarter.
    Columns: gvkeys (company IDs)
    Rows: year-quarter identifiers
    """
    records = []

    for gvkey in company_list:
        firm_data = df[df['gvkey'] == gvkey].sort_values(by='datadate')

        if n_quarters:
            firm_data = firm_data.tail(n_quarters)

        for _, row in firm_data.iterrows():
            revenue = row.get('saleq', None)
            cogs = row.get('cogsq', 0)
            cogs = 0 if pd.isna(cogs) else cogs
            sga = row.get('xsgaq', 0)
            sga = 0 if pd.isna(sga) else sga
            other_operating_expense = row.get('xoproq', 0)
            other_operating_expense = 0 if pd.isna(other_operating_expense) else other_operating_expense

            if pd.isna(revenue) or revenue == 0:
                continue

            operating_costs = cogs + sga + other_operating_expense  # + depr + interest
            ratio = operating_costs / revenue

            records.append({
                'gvkey': gvkey,
                'fyearq': row.get('fyearq'),
                'fqtr': row.get('fqtr'),
                'cost_ratio': ratio
            })

    # Create DataFrame
    cost_ratio_df = pd.DataFrame(records)

    # Pivot: rows = period, columns = gvkeys
    cost_ratio_pivot = cost_ratio_df.pivot(index=['fyearq', 'fqtr'], columns='gvkey', values='cost_ratio')

    # Optional: sort index (time) and columns (company IDs)
    cost_ratio_pivot = cost_ratio_pivot.sort_index().sort_index(axis=1)

    # Add column for average cost ratio each quarter
    cost_ratio_pivot['average'] = cost_ratio_pivot.mean(axis=1, skipna=True)

    # Add log return of average cost ratio
    cost_ratio_pivot['Log_Returns_Gamma'] = np.log(cost_ratio_pivot['average'] / cost_ratio_pivot['average'].shift(1))

    return cost_ratio_pivot


def get_T():
    """
    Time horizon in years (T)
    """   
    return 25


def get_dt():
    """
    Time step (dt)
    """
    return 1


def get_M():
    """
    Exit multiple (M)
    """
    return 10


def get_simulations():
    """
    Number of Monte Carlo runs (simulations)
    """
    return 1000000


def get_num_steps():
    """
    Number of steps in the simulation
    """
    return get_T() * 4 + 1  



def get_seasonal_factors(gvkey, df=FUNDAMENTALS, seasonal=True):
    """
    Calculates multiplicative seasonal factors for saleq for a specific company (gvkey).
    Input: gvkey (company identifier), DataFrame with columns ['fyearq', 'fqtr', 'gvkey', 'saleq']
    Returns: Dict {1: factor_Q1, 2: factor_Q2, 3: factor_Q3, 4: factor_Q4}
    """
    if not seasonal:
        return {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}  # Default seasonal factors if running in default mode

    # Filter for the specific company
    df_company = df[df['gvkey'] == gvkey].dropna(subset=['saleq'])

    if df_company.empty:
        raise ValueError(f"No data found for gvkey {gvkey}")

    # Step 1: Average revenue per quarter (for this company)
    quarterly_avg = df_company.groupby('fqtr')['saleq'].mean()

    # Step 2: Overall average revenue (for this company)
    overall_avg = df_company['saleq'].mean()

    # Step 3: Seasonal factor = quarter_avg / overall_avg
    seasonal_factors = quarterly_avg / overall_avg

    # Step 4: Normalize so that mean = 1
    seasonal_factors = seasonal_factors / seasonal_factors.mean()

    # Fill missing quarters with 1.0 if a company has missing quarters
    all_quarters = {q: seasonal_factors.get(q, 1.0) for q in [1, 2, 3, 4]}

    return all_quarters


def get_financing_cost():
    return 0.05


def get_financing_grid(gvkey):
    financing_grid = np.array([0.0, 5.0, 10.0, 20.0, 40.0])  # in millions EUR
    if gvkey not in COMPANY_LIST:
        raise ValueError(f"GVKEY {gvkey} not found in company list.")

    idx = COMPANY_LIST.index(gvkey)
    factor = COMPANY_CURRENCY_FACTORS[idx]
    adjusted_values = financing_grid * factor

    return adjusted_values

def get_C_max(gvkey):
    euro = 100
    idx = COMPANY_LIST.index(gvkey)
    factor= COMPANY_CURRENCY_FACTORS[idx]
    adjusted_value = euro * factor
    return adjusted_value


def get_revenue_cost_correlation(gvkey, df=FUNDAMENTALS):
    """
    Calculates the correlation between revenue and total cost (COGS + SG&A),
    where missing COGS or SG&A is treated as zero.
    Returns: float representing the correlation coefficient.
    """
    # Filter for the specific company and drop rows with missing revenue
    df_company = df[df['gvkey'] == gvkey].dropna(subset=['saleq'])

    if df_company.empty:
        print(f"No data found for gvkey {gvkey}, returning NaN correlation.")
        return float('nan')
    
    # Replace NaN with zero for cost components and compute total cost
    df_company['total_cost'] = df_company['cogsq'].fillna(0) + df_company['xsgaq'].fillna(0) + df_company['xoproq'].fillna(0)

    # Drop rows where total_cost is missing (in case both were NaN)
    df_company = df_company.dropna(subset=['total_cost'])

    if df_company.empty:
        print(f"No valid cost data for gvkey {gvkey}, returning NaN correlation.")
        return float('nan')
    
    # Compute correlation between revenue and total cost
    corr = df_company['saleq'].corr(df_company['total_cost'])
    return corr




if __name__ == '__main__':
    gvkey = 318456
    # [103342 SSE, 225094 VESTAS, 225597 FORTUM, 232646 ORSTED, 245628 ]NORDEX, 318456 SCATEC, 328809 NEOEN, 329260 ENCAVIS]

    print(f'Revenue (R_0): {get_R_0(gvkey)}')
    print(f'Loss Carryforward (L_0): {get_L_0(gvkey)}')
    print(f'Initial Capex (Capex_0): {get_Initial_CapEx_Ratio(gvkey)}')
    print(f'Initial Depreciation (Dep_0): {get_Depreciation_Ratio(gvkey)}')
    print(f'Initial PPE (PPE_0): {get_PPE_0(gvkey)}')
    print(f'Long-term Capex (Long_term_CAPEX): {get_Long_term_CAPEX(gvkey)}')
    print(f'eta_0: {get_eta_0(gvkey)}')
    print(f'R_f: {get_r_f()}')
    print(f'Initial growth rate (mu_0): {get_mu_0()}')
    print(f'Initial volatility of revenues (sigma_0): {get_sigma_0()}')
    # print(f'Sigma_seasonal. {get_sigma_0_seasonal()}')
    # print(f'Sigma_seasonal_mean. {get_sigma_0_seasonal_mean()}')
    print(f'Tax rate (taxrate): {get_taxrate(gvkey)}')
    print(f'Kappa mu (kappa_mu): {get_kappa_mu()}')
    print(f'Initial gamma (gamma_0): {get_gamma_0(gvkey)}')
    print(f'Gamma mean (gamma_mean): {get_gamma_mean(gvkey)}')
    print(f'Initial phi (phi_0): {get_phi_0()}')
    print(f'Phi mean (phi_mean): {get_phi_mean()}')
    print(f'Sigma market: {SIGMA_MARKET}')
    print(f'lambda_R: {get_lambda_R()}')
    print(f'lambda_mu: {get_lambda_mu()}')
    print(f'lambda_gamma: {get_lambda_gamma()}')

    print(f'seasonal_factors: {get_seasonal_factors(gvkey, FUNDAMENTALS)}')
    print(f'Financing cost: {get_financing_cost()}')
    print(f'Financing grid: {get_financing_grid(gvkey)}')
    for gvkey in COMPANY_LIST:
        print(f'correlation revenue cost {gvkey}: {get_revenue_cost_correlation(gvkey)}')
