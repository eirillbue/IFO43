# ------------------------------------------------------------
# smoke_test.py
# ------------------------------------------------------------
# Quick check that every model version in src/ still runs for all firms.
# Uses few simulation paths, so it only catches crashes and invalid output
# (NaN, missing files, broken imports). It does NOT check that the values
# are correct; compare against the baseline with 1 000 000 paths for that.
#
# Run from the repository root:  python tests/smoke_test.py
# ------------------------------------------------------------

import importlib
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)  # parameters.py reads data/ relative to the working directory

import parameters as p

N_SIMULATIONS = 2_000  # small, so the check runs in a few minutes on a GitHub runner
p.get_simulations = lambda: N_SIMULATIONS

# Model versions to test. Files that do not exist on the current branch are skipped,
# so the check works both before and after new model versions are merged.
MODEL_MODULES = [
    "NT_2025_model",
    "NT_2025_model_without_seasonality",
    "baseline_model_m1",
    "baseline_model_m2",
]

failures = []
for module_name in MODEL_MODULES:
    if importlib.util.find_spec(module_name) is None:
        print(f"SKIP {module_name} (file not found)")
        continue
    model = importlib.import_module(module_name)

    for gvkey, name in zip(p.COMPANY_LIST, p.COMPANY_NAMES):
        # Main valuation function
        try:
            v0 = model.simulate_firm_value(gvkey, save_to_file=False)
            if not np.isfinite(v0):
                failures.append(f"{module_name} {name}: V0 is not finite ({v0})")
        except Exception as error:
            failures.append(f"{module_name} {name}: simulate_firm_value raised {error!r}")
            continue

        # Sensitivity function, which needs pre-generated shocks
        rng = np.random.default_rng(gvkey)
        shape = (N_SIMULATIONS, p.get_num_steps())
        try:
            v0_s, bankruptcies = model.simulate_firm_value_sensitivity(
                gvkey,
                Z_R=rng.standard_normal(shape),
                Z_mu=rng.standard_normal(shape),
                Z_gamma=rng.standard_normal(shape),
                override_params={},
            )
            if not np.isfinite(v0_s):
                failures.append(f"{module_name} {name}: sensitivity V0 is not finite ({v0_s})")
        except Exception as error:
            failures.append(f"{module_name} {name}: simulate_firm_value_sensitivity raised {error!r}")
            continue

        print(f"OK   {module_name:36s} {name:8s} V0 = {v0:,.0f}")

if failures:
    print("\nFAILED:")
    for failure in failures:
        print(f"  {failure}")
    sys.exit(1)

print("\nAll model versions ran without errors.")
