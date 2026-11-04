"""
lightcone_paper_w18.py
=========================
Paper-quality light-cone heatmap sweep: larger system (W=18) and deeper
time range (L=1..18, matching W) than the W=8 hardware-comparison runs
(lightcone_w8_noisy_vs_noiseless.py) — a clean, ideal-circuit demonstration
of the causal structure and its saturation/boundary-reflection regime at
late L, not tied to any specific real-device run.

NOISELESS (ideal circuit) throughout, unconditionally — this is a theory
figure, not a hardware-noise comparison (see lightcone_w8_noisy_vs_noiseless.py
for that). R.NOISE_P1Q/P2Q/PM/T1/T2 are all forced to None regardless of
the module's own defaults.

Uses sweep_lightcone_probe_all (plain, non-tiled local sweep) + the same
RdPu colormap / dark cone-line convention as lightcone_w8_noisy_vs_noiseless.py.

ONLY 1 SHOT PER REALIZATION (n_shots=1) — one shot is one sample of that
realization's single quantum trajectory; with 1000 realizations already
providing the ensemble average, extra shots per realization mainly add
compute cost without changing the disorder-averaged C(x, t) much (see
lightcone_w8_noisy_vs_noiseless.py's docstring for the same reasoning).

RUNTIME -- READ BEFORE LAUNCHING
----------------------------------
Local statevector cost is exponential in W and empirically measured (this
same circuit family) at ~3.1s per realization (one pert + one unpert
circuit) at W=18, L=9 (shot count barely matters — see
front_width_scaling.py's own timing notes). Assuming roughly linear-in-L
cost (gate count per extra layer is constant, so this holds for exact
statevector simulation regardless of how entangled the state gets — no
uncontrolled blowup like MPS/tensor-network methods would have), L=1..18
sums to ~59s per realization across the full L sweep at a single p, so:

    total ~= 3 p_values * 1000 realizations * 59s ~= 49 HOURS

This is intentionally a multi-day background job (same spirit as
front_width_scaling.py and the hardware jobs left queued for a week), not
something to run interactively. checkpoint_path makes it fully resumable —
safe to interrupt and re-run the same command to pick up where it left off.

USAGE:
    nohup .butterfly/bin/python -u src/lightcone_paper_w18.py \\
        > tmp/lightcone_paper_w18.log 2>&1 &
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runs.run_disorder_avg as R

W = 18
L_VALUES = list(range(1, 19))
P_VALUES = [0.0, 0.025, 0.05]
N_REALIZATIONS = 1000
N_SHOTS = 1
BASE_SEED = 1962
ANCHOR_PARITY_AT_PERTURBATION = "odd"
CMAP = "RdPu"
CONE_LINE_COLOR = "black"

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_paper_w18_checkpoint.pkl")
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_paper_w18.png")

if __name__ == "__main__":
    R.NOISE_P1Q = None
    R.NOISE_P2Q = None
    R.NOISE_PM = None
    R.NOISE_T1 = None
    R.NOISE_T2 = None

    est_seconds_per_realization_all_L = 3.1 / 9 * sum(L_VALUES)
    est_total_hours = (
        len(P_VALUES) * N_REALIZATIONS * est_seconds_per_realization_all_L / 3600
    )

    print("=== Light-cone PAPER sweep (local, noiseless) ===")
    print(f"  W                              = {W}")
    print(f"  L_VALUES                       = {L_VALUES}")
    print(f"  P_VALUES                       = {P_VALUES}")
    print(f"  N_REALIZATIONS                 = {N_REALIZATIONS}")
    print(f"  N_SHOTS                        = {N_SHOTS}")
    print(f"  BASE_SEED                      = {BASE_SEED}")
    print(f"  ANCHOR_PARITY_AT_PERTURBATION  = {ANCHOR_PARITY_AT_PERTURBATION!r}")
    print(f"  NOISE_P1Q/P2Q/PM/T1/T2         = "
          f"{R.NOISE_P1Q}/{R.NOISE_P2Q}/{R.NOISE_PM}/{R.NOISE_T1}/{R.NOISE_T2}  (noiseless)")
    print(f"  CMAP / CONE_LINE_COLOR         = {CMAP!r} / {CONE_LINE_COLOR!r}")
    print(f"  CHECKPOINT_PATH                = {CHECKPOINT_PATH}")
    print(f"  FIGURE_PATH                    = {FIGURE_PATH}")
    print(f"  Estimated total runtime        ~= {est_total_hours:.1f} hours "
          f"(rough, linear-in-L extrapolation from W=18,L=9 timing)")
    print("=" * 64)

    stats_by_x = R.sweep_lightcone_probe_all(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_shots=N_SHOTS,
        n_realizations=N_REALIZATIONS,
        base_seed=BASE_SEED,
        device_name=None,
        checkpoint_path=CHECKPOINT_PATH,
        anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=FIGURE_PATH,
        cmap=CMAP,
        cone_line_color=CONE_LINE_COLOR,
    )
