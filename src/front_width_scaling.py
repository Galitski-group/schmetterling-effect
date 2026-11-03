"""
front_width_scaling.py
=========================
Measures the DISORDER-REALIZATION fluctuation width of the OTOC light-cone
front, w(t), and checks whether it scales as a power law w(t) ~ t^beta with
beta consistent with the KPZ universality class (beta = 1/3), the way
entanglement/operator fronts in Haar-random and Clifford circuits are known
to (Nahum, Ruhman, Vijay & Haah 2017; Nahum-Ruhman-Huse) -- versus a
diffusive-looking front (beta = 1/2) or a non-fluctuating one (beta = 0).

THEORY
------
plot_lightcone_heatmap's C(x, t) is an ENSEMBLE average over disorder
realizations from the start. That average hides that for any SINGLE
realization (one draw of the random brickwall unitary pattern + one draw
of which sites get measured, at rate p), the perturbation's front -- how
far its influence has visibly propagated -- is itself a random variable
that fluctuates realization-to-realization around the mean ballistic
trajectory x_f(t) ~ v_B * t. The KPZ scaling ansatz is

    x_f(t) = v_B * t + chi * t^beta

where chi is an O(1) random variable (converging to a Tracy-Widom
distribution for genuine KPZ growth) and beta is the roughness exponent.
The width of the front across the realization ensemble,

    w(t) = std over realizations of x_f(t)  ~  t^beta

is what this module measures: a straight line of slope beta on a log-log
plot of w vs t is the diagnostic. beta=1/3 -> KPZ; beta=1/2 would look
diffusive/random-walk-like instead; beta~0 would mean an almost sharp,
non-fluctuating front.

The p-dependence is the physically interesting part: weak measurement is
expected to eventually drive a measurement-induced phase transition (MIPT)
at some critical p_c, changing the front's universality class entirely.
p up to 0.05 here is presumably well below p_c for this chain -- the
hypothesis this module checks is whether the front STAYS KPZ-like (same
beta) with only v_B decreasing as p grows, versus beta itself drifting.

PER-REALIZATION FRONT DEFINITION
---------------------------------
Reuses the exact same circuit and per-site readout as
sweep_lightcone_probe_all (build_brickwall_lightcone_probe_all_circuit,
same flip-corrected d_r(x) = mean_shots(unpert) - mean_shots(pert) profile)
but keeps it PER REALIZATION instead of only feeding an ensemble average.
For each realization, the front position on each side of pert_site is the
farthest site whose |d_r(x)| exceeds FRONT_THRESHOLD (see _front_position).
The two directions are tracked SEPARATELY, not pooled/symmetrized: the
brickwall geometry gives one direction a one-layer head start over the
other (see build_brickwall_lightcone_probe_all_circuit's own docstring on
anchor_parity_at_perturbation, set to "odd" here for a stable, consistent
bias rather than one that flips with L's parity) -- that's a real, fixed
offset in v_B between the two sides, not a fluctuation, so it shouldn't be
averaged away; getting the same beta out of both sides independently is a
much stronger check than getting it from one.

RUNTIME -- READ BEFORE LAUNCHING
----------------------------------
Local statevector simulation cost is exponential in W and empirically
measured (this exact circuit family) at ~3.1s per realization (one pert +
one unpert circuit) at W=18, L=9 -- shot count barely matters (measured
20 vs 100 shots: same cost; the cost is circuit-build + statevector-sim,
not sampling), so N_SHOTS_PER_REALIZATION is set high (100) for a clean
per-realization threshold-crossing decision at ~zero extra cost. At the
defaults below (W=18, L=1..8, 6 p-values, target 2000 realizations/point)
total cost is ~40 HOURS -- this is intentionally a multi-day background
job (same spirit as letting hardware jobs queue for a week), not something
to run interactively. It uses AerBackend() directly (NOT
run_disorder_avg._build_backend), deliberately bypassing the module's
NOISE_* globals: this is a question about the IDEAL circuit model's
universality class, hardware noise is an unwanted confound here (and
roughly 3x slower per the same timing tests).

CHECKPOINTING
-------------
checkpoint[(W, base_seed, p, L)] = {"x_plus": [...], "x_minus": [...]},
saved after every SAVE_EVERY realizations per (p, L) point (not per
realization -- I/O overhead over thousands of iterations). Fully resumable
and INCREMENTAL: re-running with a higher n_realizations_target on a
partially-filled checkpoint collects only the difference, and
plot_front_width_scaling works on whatever's currently checkpointed at any
time, so partial progress is usable without waiting for the full run.

USAGE
-----
    # Long-running data collection (checkpoints as it goes):
    nohup .butterfly/bin/python -u src/front_width_scaling.py \\
        > tmp/front_width_scaling.log 2>&1 &

    # Just re-plot from whatever's currently checkpointed, no new circuits:
    .butterfly/bin/python src/front_width_scaling.py --plot-only
"""

import os
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import run_disorder_avg as R
from pytket.extensions.qiskit import AerBackend

W = 18
PERT_SITE = W // 2
L_VALUES = list(range(1, 9))  # stays within the uncontaminated light cone:
    # pert_site=9, W-1-pert_site=8, so L=8 is the last step before the
    # front could reach a boundary and reflect/saturate.
P_VALUES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
N_REALIZATIONS_TARGET = 2000
N_SHOTS_PER_REALIZATION = 100
FRONT_THRESHOLD = 0.4  # see module docstring: ~2.8 sigma above the
    # per-realization shot-noise floor at n_shots=100 (sigma_d = sqrt(2/n_shots)
    # ~= 0.14), scanned across ~9 sites per side, keeps the total spurious-
    # crossing rate low while sitting well below typical in-cone plateau
    # values of d_r (~0.6-1.0, see figs/lightcone_probe_all_h2_2.png).
ANCHOR_PARITY_AT_PERTURBATION = "odd"
BASE_SEED = 7
SAVE_EVERY = 25  # checkpoint save cadence, in realizations per (p, L) point

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "front_width_scaling_checkpoint.pkl")
FIGURE_PATH = str(R.FIGURE_DIR / "front_width_scaling.png")


def _front_position(
    d_r_per_site: np.ndarray, pert_site: int, threshold: float
) -> tuple[float | None, float | None]:
    """
    Farthest site on each side of pert_site whose |d_r| exceeds threshold,
    for ONE realization's flip-corrected per-site profile (index = qubit).

    Returns (x_plus, x_minus): farthest right-side x (>=1) crossing
    threshold, or None if none did; farthest left-side x (<=-1) likewise.
    """
    n = len(d_r_per_site)
    x_plus = None
    for q in range(n - 1, pert_site, -1):
        if abs(d_r_per_site[q]) > threshold:
            x_plus = q - pert_site
            break
    x_minus = None
    for q in range(0, pert_site):
        if abs(d_r_per_site[q]) > threshold:
            x_minus = q - pert_site
            break
    return x_plus, x_minus


def collect_front_width_data(
    n_realizations_target: int = N_REALIZATIONS_TARGET,
    W: int = W,
    p_values: list[float] | None = None,
    L_values: list[int] | None = None,
    pert_site: int | None = None,
    n_shots: int = N_SHOTS_PER_REALIZATION,
    threshold: float = FRONT_THRESHOLD,
    base_seed: int = BASE_SEED,
    anchor_parity_at_perturbation: str | None = ANCHOR_PARITY_AT_PERTURBATION,
    checkpoint_path: str = CHECKPOINT_PATH,
    save_every: int = SAVE_EVERY,
) -> None:
    """
    Fills checkpoint_path up to n_realizations_target front-position samples
    per (p, L) point, resuming from whatever's already there. See module
    docstring for the runtime cost and checkpoint format.
    """
    p_values = p_values if p_values is not None else P_VALUES
    L_values = L_values if L_values is not None else L_VALUES
    pert_site = pert_site if pert_site is not None else W // 2

    backend = AerBackend()
    build = R.build_brickwall_lightcone_probe_all_circuit

    checkpoint: dict[tuple, dict[str, list]] = {}
    if os.path.exists(checkpoint_path):
        checkpoint = R._load_partial(checkpoint_path)
        print(f"Loaded checkpoint <- {checkpoint_path}  ({len(checkpoint)} point(s) with existing data)")

    print(
        f"\n=== Front-width scaling (local) | W={W}  pert_site={pert_site}  "
        f"L_values={L_values}  p_values={p_values}  "
        f"target n_realizations={n_realizations_target}  n_shots={n_shots}  "
        f"threshold={threshold} ==="
    )

    t_start = time.time()
    for p in p_values:
        for L in L_values:
            ckpt_key = (W, base_seed, p, L)
            entry = checkpoint.setdefault(ckpt_key, {"x_plus": [], "x_minus": []})
            n_have = len(entry["x_plus"]) + len(entry["x_minus"])
            n_have_each = max(len(entry["x_plus"]), len(entry["x_minus"]))
            if n_have_each >= n_realizations_target:
                print(f"p={p:.3f}  L={L:2d}  [already have {n_have_each} >= target, skipping]")
                continue

            print(f"p={p:.3f}  L={L:2d}  have {n_have_each}, collecting up to "
                  f"{n_realizations_target} …")

            r = n_have_each
            while r < n_realizations_target:
                batch_end = min(r + save_every, n_realizations_target)
                for rr in range(r, batch_end):

                    def _s(axis, rr=rr, L=L, s=base_seed):
                        return (s + axis) * 1_000_000_000 + L * n_realizations_target + rr

                    shared = dict(
                        L=L, W=W, pert_site=pert_site, pert_op=R.PERT_OP, p=p,
                        seed=_s(0), init_seed=_s(1), meas_seed=_s(2), meas_seed_ud=_s(3),
                        add_barrier=False,
                        anchor_parity_at_perturbation=anchor_parity_at_perturbation,
                    )
                    qc_p = build(**shared, unperturbed=False)
                    qc_u = build(**shared, unperturbed=True)

                    sp, su = R._run_circuits(
                        [qc_p, qc_u], backend, n_shots,
                        nexus_project=R.NEXUS_PROJECT,
                        optimisation_level=R.OPTIMISATION_LEVEL, use_batch=False,
                    )
                    out_p = 1 - 2 * sp[:, 1 : 1 + W].astype(int)
                    out_u = 1 - 2 * su[:, 1 : 1 + W].astype(int)
                    d_r = out_u.mean(axis=0) - out_p.mean(axis=0)

                    x_plus, x_minus = _front_position(d_r, pert_site, threshold)
                    if x_plus is not None:
                        entry["x_plus"].append(x_plus)
                    if x_minus is not None:
                        entry["x_minus"].append(x_minus)

                r = batch_end
                checkpoint[ckpt_key] = entry
                R._save_partial(checkpoint, checkpoint_path)
                elapsed = time.time() - t_start
                print(f"  p={p:.3f}  L={L:2d}  {r}/{n_realizations_target}  "
                      f"(elapsed {elapsed/60:.1f} min)")

    print("Front-width data collection done.")


def plot_front_width_scaling(
    checkpoint_path: str = CHECKPOINT_PATH,
    p_values: list[float] | None = None,
    L_values: list[int] | None = None,
    W: int = W,
    base_seed: int = BASE_SEED,
    min_samples: int = 10,
    figure_path: str = FIGURE_PATH,
) -> None:
    """
    Loads checkpoint_path (whatever's there, complete or partial), computes
    w(L) = std over realizations of front position for each side/p, fits
    log(w) vs log(L) to get beta, and plots both against KPZ (beta=1/3) and
    diffusive (beta=1/2) reference lines. (p, L) points with fewer than
    min_samples collected realizations are skipped (too noisy a std
    estimate to be worth plotting) rather than blocking on the rest.
    """
    p_values = p_values if p_values is not None else P_VALUES
    L_values = L_values if L_values is not None else L_VALUES

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint at {checkpoint_path} — nothing to plot.")
        return
    checkpoint = R._load_partial(checkpoint_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(p_values)))

    for side, ax, key, label in (
        (0, axes[0], "x_plus", "front width  (+x side)"),
        (1, axes[1], "x_minus", "front width  (-x side)"),
    ):
        for p, color in zip(p_values, colors):
            Ls, ws = [], []
            for L in L_values:
                entry = checkpoint.get((W, base_seed, p, L))
                if entry is None:
                    continue
                samples = entry[key]
                if len(samples) < min_samples:
                    continue
                w = float(np.std(samples, ddof=1))
                if w <= 0:
                    continue
                Ls.append(L)
                ws.append(w)
            if len(Ls) < 3:
                print(f"[{label}] p={p:.3f}: only {len(Ls)} usable point(s), skipping fit/plot.")
                continue

            Ls = np.array(Ls, dtype=float)
            ws = np.array(ws, dtype=float)
            beta, log_w0 = np.polyfit(np.log(Ls), np.log(ws), 1)
            ax.plot(Ls, ws, "o", color=color, label=f"p={p:.3f}  β≈{beta:.2f}")
            fit_line = np.exp(log_w0) * Ls ** beta
            ax.plot(Ls, fit_line, "-", color=color, alpha=0.5, lw=1)

        # Reference slopes anchored at the first p-series' first plotted point,
        # for visual comparison only (not fitted to data).
        L_ref = np.array(L_values, dtype=float)
        if L_ref[0] <= 0:
            L_ref = L_ref[L_ref > 0]
        ax.plot(L_ref, L_ref ** (1 / 3) * (1.0 / L_ref[0] ** (1 / 3)),
                 "k--", lw=1, alpha=0.6, label="β=1/3 (KPZ)")
        ax.plot(L_ref, L_ref ** 0.5 * (1.0 / L_ref[0] ** 0.5),
                 "k:", lw=1, alpha=0.6, label="β=1/2 (diffusive)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t  (layers, L)")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("w(t) = std over realizations of front position")
    fig.suptitle(f"OTOC front-width scaling — W={W}")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path}")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        collect_front_width_data()
    plot_front_width_scaling()
