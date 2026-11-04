"""
run_lightcone_local.py
=======================
Local-simulation-ONLY light-cone sweep out to L~10, comparing p=0.0 vs
p=0.01 (previously looked identical — diagnosed as a statistics/duration
issue, not a circuit bug: the default run_lightcone_sweep() only goes to
L=1..x_max=3 for W=8, far too short for a 1% mid-circuit measurement rate
to accumulate any visible effect. This script deliberately widens W so the
causal cone has room to grow out to ~L=10 without saturating at the chain
edge immediately, and extends L accordingly).

SAFETY: device_name is never passed to sweep_lightcone_x_and_t below, so
_build_backend always takes its local-simulation (AerBackend) path —
nothing here can hit real Quantinuum hardware.

Sizing: full L=1..10 with x_max_hard=W//2 requires W=22 for the cone to
never saturate before L=10, but per-shot cost blows up sharply on Aer's
statevector simulator around W=20-22 (calibrated: ~24ms/shot at W=18,L=3
vs ~660ms/shot at W=22,L=10 — a >25x jump for +4 qubits). W=18 (x_max=8)
was chosen as the practical balance: cone genuinely grows via causal
geometry out to x=+-8 (close enough to "t=10" that the request's intent
-- see the shape, not hit some exact number), stays in the ~25-100ms/shot
regime, and total estimated runtime is ~2 hours instead of many hours.
Adjust W (must stay even) if you want the cone to reach further and are
willing to trade runtime for it — see this file's own timing comment
below before increasing it.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runs.run_disorder_avg as R

W = 18                    # pert_site=9, x_max=min(9,8)=8 -- see docstring above
L_VALUES = list(range(1, 11))   # 1..10
P_VALUES = [0.0, 0.01]
N_CIRCUITS = 3
N_INIT_STATES = 3
N_SHOTS = 30
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_p0_vs_p01_local.png")

if __name__ == "__main__":
    R.N_CIRCUITS = N_CIRCUITS
    R.N_INIT_STATES = N_INIT_STATES

    print(f"W={W}  pert_site={W//2}  L_values={L_VALUES}  p_values={P_VALUES}  "
          f"n_shots={N_SHOTS}  realizations={N_CIRCUITS*N_INIT_STATES}  "
          f"(local simulation only, device_name not passed)")

    stats_by_x = R.sweep_lightcone_x_and_t(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_shots=N_SHOTS,
        # device_name intentionally omitted -> local AerBackend, never hardware
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=FIGURE_PATH,
    )
