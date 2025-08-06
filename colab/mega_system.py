# %% [markdown]
"""
# Mega System Colab

This notebook orchestrates the **entire** codebase through the `mega_system.MegaSystem` class.
"""

# %%
"""Ensure project root is on the Python path."""
import pathlib, sys, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Project root added to PYTHONPATH")

# %%
from mega_system import MegaSystem

system = MegaSystem()
report = system.run_cycle({"message": "hello world"})
print(json.dumps(report, indent=2))