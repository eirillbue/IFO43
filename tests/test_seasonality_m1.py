# ------------------------------------------------------------
# test_seasonality_m1.py
# ------------------------------------------------------------
# Checks that seasonality in model version M1 (src/baseline_model_m1.py) is a
# pure level adjustment: with no growth and no volatility, revenue must return
# to its starting value after every full year, R(4k) = R_0.
#
# Growth and volatility are switched off by overriding the parameter getters in
# memory (mu_0 = mu_mean = 0, sigma_0 = sigma_mean = 0, eta_0 = 0). The
# exponent in equation (27) is then zero, so revenue changes only through the
# seasonal ratio S(t+dt)/S(t), which multiplies to 1 over four quarters.
# No model or parameter files are changed.
#
# Run from the repository root:  python tests/test_seasonality_m1.py
# ------------------------------------------------------------

import importlib
import os
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)  # parameters.py reads data/ relative to the working directory

import parameters as p


def _zero(*args, **kwargs):
    return 0.0


def simulate_revenue(module_name, gvkey):
    """
    Simulate one revenue path with no growth and no volatility, and return it.

    The model only returns V0, so the full results dictionary is captured by
    replacing pickle.dump in the model module, and the (empty) output file the
    model creates is written to a temporary directory.
    """
    # One path, no growth (mu, and eta so mu stays at zero), no volatility (sigma)
    overrides = {
        "get_simulations": lambda: 1,
        "get_mu_0": _zero,
        "get_mu_mean": _zero,
        "get_sigma_0": _zero,
        "get_sigma_mean": _zero,
        "get_eta_0": _zero,
    }
    originals = {name: getattr(p, name) for name in overrides}
    for name, function in overrides.items():
        setattr(p, name, function)

    model = importlib.import_module(module_name)
    captured = []
    original_pickle = model.pickle
    model.pickle = types.SimpleNamespace(dump=lambda obj, f: captured.append(obj))

    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            model.simulate_firm_value(gvkey, save_to_file=True)
    finally:
        os.chdir(cwd)
        model.pickle = original_pickle
        for name, function in originals.items():
            setattr(p, name, function)

    return captured[0]["results"]["R"][0]  # the single path, shape (num_steps,)


def test_revenue_returns_to_start_every_year():
    for gvkey, name in zip(p.COMPANY_LIST, p.COMPANY_NAMES):
        revenue = simulate_revenue("baseline_model_m1", gvkey)
        start = revenue[0]
        year_ends = revenue[4::4]  # R(4), R(8), ..., R(100)
        max_relative_error = np.max(np.abs(year_ends / start - 1))
        print(f"{name:8s} R0 = {start:12,.2f}   R(4k) for k = 1..{len(year_ends)}: "
              f"max relative deviation {max_relative_error:.1e}")
        assert np.allclose(year_ends, start, rtol=1e-10, atol=0), (
            f"{name}: revenue does not return to R0 after a full year "
            f"(max relative deviation {max_relative_error:.3e})"
        )


if __name__ == "__main__":
    test_revenue_returns_to_start_every_year()
    print("\nOK: in M1, revenue returns to R0 after every fourth quarter for all firms.")
