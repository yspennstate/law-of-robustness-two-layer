# Numerical checks for "A law of robustness for two-layer neural networks with arbitrary weights"

This repository holds the numerical scripts accompanying the paper by Yitzchak Shmalo.
Each script reproduces one of the checks reported in the paper's numerics section; all
use the fixed seed 20260707 and run in well under a minute. Run any of them with
`python numerics/<name>.py`.

- `check_rigidity.py` — the rigidity bound |alpha_j| <= 2L on random canonical ReLU networks, including cancellation plants.
- `check_sphere_cap.py` — the spherical factor sqrt(1-t^2) is exact.
- `check_d2_rigidity_fails.py` — the four-unit cycle showing rigidity fails on the circle (d=2).
- `check_upper_bound.py` — the width-2n interpolant with Lipschitz constant sqrt(7).
- `check_law_fast.py`, `check_law_empirical.py` — trained networks satisfy the lower bound.
- `validate_at_scale.py` — the rigidity and construction checks at larger scale (with a saved log).
- `check_static_rigidity.py` — single-pole static rigidity.
- `check_projection_floor.py` — the width-1 projection-capacity floor.
- `check_sector_throttle.py` — the deep/shallow/mid sector multiplicity throttles.

Requires only NumPy.
