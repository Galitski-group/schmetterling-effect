"""
self_averaging_scan.py
========================
Scans Var[C] / |<C>|^2 -- the squared coefficient of variation, across
disorder realizations, of a spatially-averaged causal-cone observable --
across system size N in [8, 10, 12, 14, 16] (even only: the brickwall
circuit requires even W), for p in P_VALUES (currently a 6-point sweep
from 0.0 to 0.05).

L (time depth) is P-DEPENDENT, not a single fixed ratio: p=0.0 uses
L = N // 2, any p>0 uses L = N // 4. This asymmetry exists because the
unitary (p=0) correlator stays large enough to probe meaningfully all the
way to N//2, but a measured (p>0) correlator decays fast enough that
by N//2 its mean <C> has collapsed toward 0 faster than Var[C] has,
inflating the ratio for a reason that has nothing to do with self-
averaging (see the run before this one). Using the shallower N//4 for
p>0 probes it before that decay sets in, at the cost of a shallower
depth than the p=0 curve -- p=0 is intentionally NOT probed at the same
absolute L as the rest, only at the same fraction-of-N-appropriate-to-
that-regime.

n_realizations is SHARED across every p at a given N (see
_n_realizations_for) -- ALPHA * N * (N//2), using the p=0 depth as the
single reference regardless of which p the circuit itself runs at, so
every p gets the same statistical power at a given N.

PER-REALIZATION OBSERVABLE: for realization r, C_r = mean over sites x
with |x| <= L//4 on each side of pert_site (a conservative sub-region
well inside the causal cone |x| <= L, so every included site has
unambiguously been reached, using whichever L applies for that p) of
d_r(x) -- the same flip-corrected pert-vs-unpert difference used
everywhere else in this project. Spatially averaging within the cone
(rather than reading a single site) gives a more representative "bulk"
value per realization instead of hinging on one arbitrarily-chosen site
-- though note this window is narrow (even 0, i.e. just pert_site itself)
for small N, since L//4 of an already-small L rounds down fast; real
multi-site averaging only kicks in at larger N.
The reported statistics are THEN taken across realizations: C = mean_r(C_r),
var = Var_r(C_r), ratio = var / C^2.

This ratio is a self-averaging diagnostic: if it vanishes as N grows, the
OTOC signal concentrates around its mean in the thermodynamic limit
(self-averaging); if it plateaus at some nonzero value instead, disorder-
to-disorder fluctuations stay O(1) even for large systems.

n_realizations SCALES AS N*L per (N, p) point (per the original request)
-- note this means larger N costs more in TWO compounding ways: per-
realization circuit cost grows exponentially in N (statevector size), AND
n_realizations itself grows with N (quadratically for p=0.0, since
L=N//2; more slowly for p=0.05, since L=N//4). Both effects are folded
into the printed time estimate below -- read it before launching the
full run.

NOISELESS throughout -- a question about the ideal circuit model's
self-averaging property, not hardware noise (same reasoning as
front_width_scaling.py / lightcone_paper_w18.py).

N_SHOTS=100, NOT 1 -- unlike the other local sweeps in this project, shot
count matters here: we are measuring Var[C_r] directly, so shot noise
riding on top of the true disorder-driven variance would inflate the very
quantity this script exists to measure, unless suppressed with enough
shots per realization. Shot count is nearly free time-wise regardless
(cost is circuit-build + statevector-sim, not sampling -- see
front_width_scaling.py's own timing notes), so there's no real tradeoff
in choosing it generously.

USAGE:
    # Time estimate only -- calibrates on real (N, L) points, prints a
    # projected total, submits nothing:
    .butterfly/bin/python src/self_averaging_scan.py --estimate-only

    # Full run (checkpointed/resumable per (N, p) point):
    nohup .butterfly/bin/python -u src/self_averaging_scan.py \\
        > tmp/self_averaging_scan.log 2>&1 &
"""

import os
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import run_disorder_avg as R
from pytket.extensions.qiskit import AerBackend

N_VALUES = [8, 10, 12, 14, 16]
P_VALUES = [0.0, 0.0125, 0.025, 0.05]
N_SHOTS = 100
ANCHOR_PARITY_AT_PERTURBATION = "odd"
BASE_SEED = 1962
ALPHA = 5  # realization-count multiplier, see _n_realizations_for

CHECKPOINT_PATH = str(
    pathlib.Path(R.PARTIALS_DIR) / "self_averaging_scan_checkpoint.pkl"
)
FIGURE_PATH = str(R.FIGURE_DIR / "self_averaging_scan.png")


def _L_for(N: int, p: float) -> int:
    return N // 2 if p == 0.0 else N // 4


def _n_realizations_for(N: int) -> int:
    """SHARED across both p values -- ALPHA * N * (N//2), using the
    p=0.0 depth (N//2) as the single reference L regardless of which p
    the circuit itself actually runs at, so both curves get the same
    statistical power at a given N rather than p=0.05 (shallower L)
    getting fewer realizations than p=0.0 by construction."""
    return ALPHA * N * (N // 2)


def _cone_sites(N: int, L: int, pert_site: int) -> list[int]:
    """Qubit indices q with |q - pert_site| <= L // 4, clipped to [0, N-1]."""
    half_width = L // 4
    lo = max(0, pert_site - half_width)
    hi = min(N - 1, pert_site + half_width)
    return list(range(lo, hi + 1))


def _time_one_realization(N: int, L: int, p: float, n_shots: int) -> float:
    pert_site = N // 2
    backend = AerBackend()
    build = R.build_brickwall_lightcone_probe_all_circuit

    t0 = time.time()
    shared = dict(
        L=L,
        W=N,
        pert_site=pert_site,
        pert_op=R.PERT_OP,
        p=p,
        seed=1,
        init_seed=2,
        meas_seed=3,
        meas_seed_ud=4,
        add_barrier=False,
        anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
    )
    qc_p = build(**shared, unperturbed=False)
    qc_u = build(**shared, unperturbed=True)
    R._run_circuits(
        [qc_p, qc_u],
        backend,
        n_shots,
        nexus_project=R.NEXUS_PROJECT,
        optimisation_level=R.OPTIMISATION_LEVEL,
        use_batch=False,
    )
    return time.time() - t0


def estimate(n_shots: int = N_SHOTS) -> float:
    """Calibrates real per-realization cost at each (N, L=N//2) point (one
    timing per p, since measurement-pattern density depends on p too), then
    projects the full sweep's total wall-clock time. Returns that total in
    seconds; prints a full breakdown table as it goes."""
    print(f"{'N':>4} {'L':>4} {'p':>6} {'n_real':>8} {'s/real':>10} {'proj (s)':>12}")
    grand_total = 0.0
    for N in N_VALUES:
        for p in P_VALUES:
            L = _L_for(N, p)
            n_real = _n_realizations_for(N)
            t_one = _time_one_realization(N, L, p, n_shots)
            proj = t_one * n_real
            grand_total += proj
            print(f"{N:>4} {L:>4} {p:>6.3f} {n_real:>8} {t_one:>10.3f} {proj:>12.1f}")
    print(
        f"\nProjected total: {grand_total:.1f}s = {grand_total/60:.1f} min = "
        f"{grand_total/3600:.2f} hr"
    )
    return grand_total


def run_scan(n_shots: int = N_SHOTS, checkpoint_path: str = CHECKPOINT_PATH) -> dict:
    """Fills checkpoint_path with {(N, p): {"C":.., "var":.., "ratio":.., "n":.., "cone_sites":..}},
    resuming any (N, p) points already present. Returns the full dict."""
    checkpoint: dict[tuple, dict] = {}
    if os.path.exists(checkpoint_path):
        checkpoint = R._load_partial(checkpoint_path)
        print(
            f"Loaded checkpoint <- {checkpoint_path}  ({len(checkpoint)} point(s) already done)"
        )

    for N in N_VALUES:
        pert_site = N // 2
        backend = AerBackend()
        build = R.build_brickwall_lightcone_probe_all_circuit

        for p in P_VALUES:
            L = _L_for(N, p)
            n_real = _n_realizations_for(N)
            cone_sites = _cone_sites(N, L, pert_site)
            key = (N, p)
            if key in checkpoint and checkpoint[key]["n"] == n_real:
                print(
                    f"N={N:3d}  L={L:3d}  p={p:.3f}  [already have {n_real}, skipping]"
                )
                continue

            print(
                f"N={N:3d}  L={L:3d}  p={p:.3f}  n_realizations={n_real}  "
                f"cone_sites={cone_sites} …"
            )
            C_r_vals = []
            for r in range(n_real):

                def _s(axis, r=r, L=L, s=BASE_SEED):
                    return (s + axis) * 1_000_000_000 + L * n_real + r

                shared = dict(
                    L=L,
                    W=N,
                    pert_site=pert_site,
                    pert_op=R.PERT_OP,
                    p=p,
                    seed=_s(0),
                    init_seed=_s(1),
                    meas_seed=_s(2),
                    meas_seed_ud=_s(3),
                    add_barrier=False,
                    anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
                )
                qc_p = build(**shared, unperturbed=False)
                qc_u = build(**shared, unperturbed=True)
                sp, su = R._run_circuits(
                    [qc_p, qc_u],
                    backend,
                    n_shots,
                    nexus_project=R.NEXUS_PROJECT,
                    optimisation_level=R.OPTIMISATION_LEVEL,
                    use_batch=False,
                )
                # bits 1..N are the per-site readouts (bit 0 is scratch).
                out_p = 1 - 2 * sp[:, 1 : 1 + N].astype(int)
                out_u = 1 - 2 * su[:, 1 : 1 + N].astype(int)
                d_r_per_site = out_u.mean(axis=0) - out_p.mean(axis=0)
                C_r_vals.append(float(d_r_per_site[cone_sites].mean()))

            C_r_vals = np.array(C_r_vals)
            C = float(C_r_vals.mean())
            var = float(C_r_vals.var(ddof=1))
            ratio = var / abs(C) ** 2 if C != 0 else float("inf")
            checkpoint[key] = {
                "C": C,
                "var": var,
                "ratio": ratio,
                "n": n_real,
                "cone_sites": cone_sites,
            }
            R._save_partial(checkpoint, checkpoint_path)
            print(f"  -> C={C:+.4f}  var={var:.4f}  var/C^2={ratio:.4f}")

    return checkpoint


def plot_scan(checkpoint: dict, figure_path: str = FIGURE_PATH) -> None:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdPu")
    n = len(P_VALUES)
    colors = {
        p: cmap(0.9 - 0.5 * (i / max(1, n - 1)))
        for i, p in enumerate(sorted(P_VALUES))
    }  # lowest p -> dark purple, highest p -> lighter pink (same convention
       # as plot_h2_2_vs_tm_styles.py), works for any number of p values.
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for p in P_VALUES:
            Ns = [N for N in N_VALUES if (N, p) in checkpoint]
            ratios = [checkpoint[(N, p)]["ratio"] for N in Ns]
            ax.plot(Ns, ratios, "o-", color=colors[p], label=f"p={p:.3f}")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$\mathrm{Var}[C]/|\langle C\rangle|^2$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
    print(f"Figure saved → {figure_path}")


if __name__ == "__main__":
    if "--estimate-only" in sys.argv:
        estimate()
    else:
        ckpt = run_scan()
        plot_scan(ckpt)
