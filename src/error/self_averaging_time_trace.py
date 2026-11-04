"""
self_averaging_time_trace.py
===============================
Companion to self_averaging_scan.py: instead of scanning system size N at
a fixed depth ratio, this FIXES N=16 and traces Var[C] / |<C>|^2 as a
function of circuit depth L itself, L = 0, 2, 4, ..., MAX_L (MAX_L = N),
for every p in P_VALUES.

n_realizations = ALPHA * N * MAX_L, ONE SHARED count for every (L, p)
point in the whole trace -- not re-derived per L or per p, so every point
on every curve has identical statistical power and only L itself moves.

PER-REALIZATION OBSERVABLE: same as self_averaging_scan.py -- for
realization r, C_r = mean over sites x with |x| <= L//4 on each side of
pert_site (clipped to [0, N-1]) of d_r(x). At L=0 this window collapses
to just pert_site itself (half_width = 0), and since neither the forward
nor backward evolution has run yet, C_r there is deterministic (the bare
effect of the perturbation gate, no circuit to scramble it) -- Var[C]
should come out ~0 at L=0 by construction, a useful trivial checkpoint
for the trace rather than a bug.

NOISELESS throughout, N_SHOTS=100 (not 1) -- same reasoning as
self_averaging_scan.py: this is a question about the ideal circuit
model, and shot noise would contaminate the very variance being measured
otherwise (shot count is nearly free time-wise regardless).

USAGE:
    # Time estimate only:
    .butterfly/bin/python src/self_averaging_time_trace.py --estimate-only

    # Full run (checkpointed/resumable per (L, p) point):
    nohup .butterfly/bin/python -u src/self_averaging_time_trace.py \\
        > tmp/self_averaging_time_trace.log 2>&1 &
"""

import os
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import runs.run_disorder_avg as R
from pytket.extensions.qiskit import AerBackend

N = 16
MAX_L = 12
L_VALUES = list(range(0, MAX_L + 1, 1))
P_VALUES = [0.0, 0.0125, 0.025, 0.05]
ALPHA = 5
N_REALIZATIONS = ALPHA * N * MAX_L
N_SHOTS = 100
ANCHOR_PARITY_AT_PERTURBATION = "odd"
BASE_SEED = 1962

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "self_averaging_time_trace_checkpoint.pkl")
FIGURE_PATH = str(R.FIGURE_DIR / "self_averaging_time_trace.png")


def _cone_sites(L: int, pert_site: int) -> list[int]:
    """Qubit indices q with |q - pert_site| <= L // 4, clipped to [0, N-1]."""
    half_width = L // 4
    lo = max(0, pert_site - half_width)
    hi = min(N - 1, pert_site + half_width)
    return list(range(lo, hi + 1))


def _time_one_realization(L: int, p: float, n_shots: int) -> float:
    pert_site = N // 2
    backend = AerBackend()
    build = R.build_brickwall_lightcone_probe_all_circuit

    t0 = time.time()
    shared = dict(
        L=L, W=N, pert_site=pert_site, pert_op=R.PERT_OP, p=p,
        seed=1, init_seed=2, meas_seed=3, meas_seed_ud=4,
        add_barrier=False, anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
    )
    qc_p = build(**shared, unperturbed=False)
    qc_u = build(**shared, unperturbed=True)
    R._run_circuits([qc_p, qc_u], backend, n_shots, nexus_project=R.NEXUS_PROJECT,
                     optimisation_level=R.OPTIMISATION_LEVEL, use_batch=False)
    return time.time() - t0


def estimate(n_shots: int = N_SHOTS) -> float:
    print(f"N={N}  MAX_L={MAX_L}  L_VALUES={L_VALUES}  P_VALUES={P_VALUES}  "
          f"N_REALIZATIONS={N_REALIZATIONS} (shared)")
    print(f"{'L':>4} {'p':>6} {'s/real':>10} {'proj (s)':>12}")
    grand_total = 0.0
    for L in L_VALUES:
        for p in P_VALUES:
            t_one = _time_one_realization(L, p, n_shots)
            proj = t_one * N_REALIZATIONS
            grand_total += proj
            print(f"{L:>4} {p:>6.4f} {t_one:>10.3f} {proj:>12.1f}")
    print(f"\nProjected total: {grand_total:.1f}s = {grand_total/60:.1f} min = "
          f"{grand_total/3600:.2f} hr")
    return grand_total


def run_trace(n_shots: int = N_SHOTS, checkpoint_path: str = CHECKPOINT_PATH) -> dict:
    """Fills checkpoint_path with {(L, p): {"C":.., "var":.., "ratio":.., "n":.., "cone_sites":..}}."""
    checkpoint: dict[tuple, dict] = {}
    if os.path.exists(checkpoint_path):
        checkpoint = R._load_partial(checkpoint_path)
        print(f"Loaded checkpoint <- {checkpoint_path}  ({len(checkpoint)} point(s) already done)")

    pert_site = N // 2
    build = R.build_brickwall_lightcone_probe_all_circuit

    for L in L_VALUES:
        cone_sites = _cone_sites(L, pert_site)
        backend = AerBackend()

        for p in P_VALUES:
            key = (L, p)
            if key in checkpoint and checkpoint[key]["n"] == N_REALIZATIONS:
                print(f"L={L:3d}  p={p:.4f}  [already have {N_REALIZATIONS}, skipping]")
                continue

            print(f"L={L:3d}  p={p:.4f}  n_realizations={N_REALIZATIONS}  "
                  f"cone_sites={cone_sites} …")
            C_r_vals = []
            for r in range(N_REALIZATIONS):
                def _s(axis, r=r, L=L, s=BASE_SEED):
                    return (s + axis) * 1_000_000_000 + L * N_REALIZATIONS + r

                shared = dict(
                    L=L, W=N, pert_site=pert_site, pert_op=R.PERT_OP, p=p,
                    seed=_s(0), init_seed=_s(1), meas_seed=_s(2), meas_seed_ud=_s(3),
                    add_barrier=False, anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
                )
                qc_p = build(**shared, unperturbed=False)
                qc_u = build(**shared, unperturbed=True)
                sp, su = R._run_circuits(
                    [qc_p, qc_u], backend, n_shots, nexus_project=R.NEXUS_PROJECT,
                    optimisation_level=R.OPTIMISATION_LEVEL, use_batch=False,
                )
                out_p = 1 - 2 * sp[:, 1 : 1 + N].astype(int)
                out_u = 1 - 2 * su[:, 1 : 1 + N].astype(int)
                d_r_per_site = out_u.mean(axis=0) - out_p.mean(axis=0)
                C_r_vals.append(float(d_r_per_site[cone_sites].mean()))

            C_r_vals = np.array(C_r_vals)
            C = float(C_r_vals.mean())
            var = float(C_r_vals.var(ddof=1))
            ratio = var / abs(C) ** 2 if C != 0 else float("inf")
            checkpoint[key] = {"C": C, "var": var, "ratio": ratio, "n": N_REALIZATIONS,
                                "cone_sites": cone_sites}
            R._save_partial(checkpoint, checkpoint_path)
            print(f"  -> C={C:+.4f}  var={var:.4f}  var/C^2={ratio:.4f}")

    return checkpoint


def plot_trace(checkpoint: dict, figure_path: str = FIGURE_PATH, stat: str = "ratio",
                skip_L0: bool = True) -> None:
    """stat: "ratio" (Var[C]/|<C>|^2) or "var" (Var[C] itself). skip_L0=True
    drops the trivial L=0 point (deterministic, Var=0 by construction --
    see module docstring) from the plot; the checkpoint still has it."""
    import matplotlib.pyplot as plt

    ylabels = {
        "ratio": r"$\mathrm{Var}[C]/|\langle C\rangle|^2$",
        "var": r"$\mathrm{Var}[C]$",
    }
    cmap = plt.get_cmap("RdPu")
    n = len(P_VALUES)
    colors = {
        p: cmap(0.9 - 0.5 * (i / max(1, n - 1)))
        for i, p in enumerate(sorted(P_VALUES))
    }
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for p in P_VALUES:
            Ls = [L for L in L_VALUES if (L, p) in checkpoint and not (skip_L0 and L == 0)]
            vals = [checkpoint[(L, p)][stat] for L in Ls]
            ax.plot(Ls, vals, "o-", color=colors[p], label=f"p={p:.4f}")
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(ylabels[stat])
        ax.set_title(f"N={N}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
    print(f"Figure saved → {figure_path}")


if __name__ == "__main__":
    if "--estimate-only" in sys.argv:
        estimate()
    else:
        ckpt = run_trace()
        plot_trace(ckpt)
