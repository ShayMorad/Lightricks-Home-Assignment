"""
edit_image.py – Command‑line front‑end for the *edit‑image* toolchain.

Responsibilities
----------------
1. Parse **`--config / -c`** argument pointing to a JSON configuration
   file.
2. Delegate validation to :pyfunc:`JSON_Loader.load_config`.
3. Load the requested input image (*PNG/JPG*) with *Matplotlib*.
4. Run the ordered operation pipeline via :pyfunc:`Filters.apply_operations`.
5. Save the result and/or show a preview window depending on config flags.

Example
~~~~~~~
```bash
python edit-image.py --config test1.json
```
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt

# module1: schema + logic validation
from JSON_Loader import load_config

# module2: image processing
from Filters import Filters

# ---------------------------------------------------------------------------
# High‑level workflow helper
# ---------------------------------------------------------------------------
def run_operations(cfg_path: str) -> None:
    """End‑to‑end execution for one configuration file."""

    # 1. Parse + validate JSON configuration ---------------------------------
    cfg = load_config(cfg_path)

    # 2. Read the input image --------------------------------------------------
    img = plt.imread(cfg["input"])  # supports PNG/JPG via matplotlib

    # 3. Apply the operation pipeline -----------------------------------------
    result = Filters.apply_operations(img, cfg["operations"])

    # 4. Save and/or display as requested -------------------------------------
    if cfg.get("output"):
        Filters.save(result, cfg["output"])

    if cfg.get("display"):
        Filters.show(result)


def main() -> None:
    """Entry‑point used by`python edit-image.py`."""
    parser = argparse.ArgumentParser(prog="edit-image",
                                     description="Apply image‑editing pipeline from a JSON config file.")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    run_operations(args.config)


if __name__ == "__main__":
    main()
