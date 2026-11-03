"""
lightcone_w8_noisy_vs_noiseless.py
=====================================
Local-simulation companion to the real H2-1 hardware light-cone sweep
(run_lightcone_h2_2_tiled.py / figs/lightcone_probe_all_h2_2.png): same
W=8, odd anchor parity, BASE_SEED=1962 (identical realizations/circuits to
the hardware run, modulo tiling), L=1..8, n_realizations=180, p=0.0/0.025/
0.05 — but run TWICE locally, once through AerBackend's calibrated H2 noise
model (NOISE_P1Q/P2Q/PM — see run_disorder_avg.py's own comments on where
those numbers come from) and once fully noiseless (ideal circuit), so the
three can be compared side by side: real hardware vs. "hardware noise
model but no other real-device effects" vs. "the circuit's ideal answer."

ONLY 1 SHOT PER REALIZATION (n_shots=1), unlike sweep_lightcone_probe_all's
usual N_SHOTS=100 default — deliberately matching how a single realization
is a single quantum trajectory (one shot IS one sample of that trajectory,
see the front-width-scaling module's discussion of this), and because with
180 realizations already providing the ensemble average, extra shots per
realization mainly just add compute cost without changing the disorder-
averaged C(x, t) much. Cheap either way at W=8 (unlike the W=18 front-width
study, an 8-qubit statevector sim is ~instant).

Uses sweep_lightcone_probe_all (the plain, non-tiled local sweep) directly
— no hardware submission machinery needed here.

USAGE (each condition gets its own log and checkpoint/figure; run both,
in parallel if you like, since they're independent local processes):
    nohup .butterfly/bin/python -u src/lightcone_w8_noisy_vs_noiseless.py --noiseless \\
        > tmp/lightcone_w8_noiseless.log 2>&1 &
    nohup .butterfly/bin/python -u src/lightcone_w8_noisy_vs_noiseless.py --noisy \\
        > tmp/lightcone_w8_noisy.log 2>&1 &

Re-running the same command later is safe/resumable — sweep_lightcone_probe_all's
own checkpoint_path skip-if-done logic applies per (p, L) point.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_disorder_avg as R

W = 8
L_VALUES = list(range(1, 9))
P_VALUES = [0.0, 0.025, 0.05]
N_REALIZATIONS = 180
N_SHOTS = 1
BASE_SEED = 1962  # same seed as the real H2-1 hardware run — identical
    # per-realization circuits (see sweep_lightcone_probe_all's seed
    # formula), so this is a directly comparable baseline, not just a
    # same-shape one.
ANCHOR_PARITY_AT_PERTURBATION = "odd"
CMAP = "RdPu"
CONE_LINE_COLOR = "black"  # RdPu puts 0 at the PALE end (unlike viridis),
    # so the cone-edge lines need a dark color to stay visible there.

if __name__ == "__main__":
    if "--noisy" in sys.argv:
        mode = "noisy"
    elif "--noiseless" in sys.argv:
        mode = "noiseless"
    else:
        raise SystemExit(
            "Pass exactly one of --noisy or --noiseless, e.g.:\n"
            "    .butterfly/bin/python src/lightcone_w8_noisy_vs_noiseless.py --noiseless"
        )

    if mode == "noiseless":
        R.NOISE_P1Q = None
        R.NOISE_P2Q = None
        R.NOISE_PM = None
        R.NOISE_T1 = None
        R.NOISE_T2 = None
    # else "noisy": leave R.NOISE_P1Q/P2Q/PM/T1/T2 at their module defaults
    # (the calibrated H2 noise model already configured at the top of
    # run_disorder_avg.py) — _build_backend picks them up automatically.

    checkpoint_path = str(pathlib.Path(R.PARTIALS_DIR) / f"lightcone_w8_{mode}_checkpoint.pkl")
    figure_path = str(R.FIGURE_DIR / f"lightcone_w8_{mode}.png")

    print(f"=== Light-cone local sweep ({mode}) | W={W}  L_values={L_VALUES}  "
          f"p_values={P_VALUES}  n_realizations={N_REALIZATIONS}  n_shots={N_SHOTS}  "
          f"anchor={ANCHOR_PARITY_AT_PERTURBATION!r}  base_seed={BASE_SEED} ===")
    print(f"NOISE_P1Q={R.NOISE_P1Q}  NOISE_P2Q={R.NOISE_P2Q}  NOISE_PM={R.NOISE_PM}  "
          f"NOISE_T1={R.NOISE_T1}  NOISE_T2={R.NOISE_T2}")

    stats_by_x = R.sweep_lightcone_probe_all(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_shots=N_SHOTS,
        n_realizations=N_REALIZATIONS,
        base_seed=BASE_SEED,
        device_name=None,
        checkpoint_path=checkpoint_path,
        anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=figure_path,
        cmap=CMAP,
        cone_line_color=CONE_LINE_COLOR,
    )
