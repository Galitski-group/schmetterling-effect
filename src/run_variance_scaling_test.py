"""
run_variance_scaling_test.py
=============================
Empirical scan of how per-realization disorder variance (of d_r, the
±1-encoded perturbed-vs-unperturbed difference) scales with system size W,
to project how many realizations a real W=54 run would need for a given
target precision. Local-simulation-ONLY, noiseless, p=0.0, n_shots=1 per
circuit (matching the real single-shot-per-circuit tiled hardware plan —
realizations, not native shot multiplicity, provide the statistics; using
the default n_shots=100 here would average down genuine single-shot
measurement noise and understate the variance a real run will actually see).

W=54 cannot be simulated locally at all (2^54 amplitudes — hundreds of
petabytes), even setting aside AerBackend's own 40-qubit default cap — so
this measures the TREND across the largest W still cheap to simulate
directly (W=6..18) and extrapolates to W=54, rather than measuring at
W=54 itself.

Two L strategies per W, since "how does variance behave" depends on WHICH
physical regime you're asking about:
  - L_FIXED=4      : same absolute depth at every W — probes whether
                     variance has any intrinsic W-dependence separate from
                     finite-size/boundary effects (light cone hasn't
                     reached the edges yet for any W in this range at L=4).
  - L_SCALED=W//2   : cone just reaching the system boundary — the
                     "fully-spread," most-relevant-to-a-real-run regime.

At each (W, L) point, n_realizations=500 gives a precise estimate of the
TRUE per-realization variance (via compute_C_t's own "var" output — the
sample variance of d_r across all 500 realizations). Realizations are
i.i.d. by construction (independent Haar-random gate/init/measurement
seeds), so se(n) = sqrt(var/n) for ANY target n follows analytically from
that one variance estimate — no need to actually re-run at smaller n or
bootstrap.

"Bulk" sites are averaged per point: |x| <= L-2 (excludes the perturbation
site itself, x=0, whose early-time behavior is a distinct deterministic-
tip effect not representative of typical propagated-signal variance; and
excludes the outermost 1-2 sites near the causal edge, where variance
behaves qualitatively differently right at the threshold where the front
has just arrived) — this isolates the variance of genuinely-propagated,
already-thermalizing bulk signal, which is what a real experiment's
precision budget actually needs to resolve.

Output: two plots —
  1. bulk-averaged Var(d_r) vs W, for both L strategies.
  2. implied n_realizations needed for se=0.02 vs W (n = var / se^2),
     with a simple fit extrapolated to W=54.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import run_disorder_avg as R

R.NOISE_P1Q = None
R.NOISE_P2Q = None
R.NOISE_PM = None
R.NOISE_T1 = None
R.NOISE_T2 = None

W_VALUES = [6, 8, 10, 12, 14, 16, 18]
L_FIXED = 4
N_REALIZATIONS = 500
P = 0.0
ANCHOR = "even"
TARGET_SE = 0.02
W_PROJECT = 54
FIGURE_PATH = str(R.FIGURE_DIR / "variance_scaling_test.png")
RESULTS_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "variance_scaling_test_results.pkl")


def bulk_var(stats_by_x, L, p=P):
    """Average Var(d_r) over bulk sites |x| <= L-2, x != 0."""
    vals = []
    for x, s in stats_by_x.items():
        if x == 0 or abs(x) > max(L - 2, 0):
            continue
        if L in s["var"].get(p, {}):
            vals.append(s["var"][p][L])
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


if __name__ == "__main__":
    results = {}  # (W, "fixed"|"scaled") -> (L, var, n_bulk_sites)

    for W in W_VALUES:
        for label, L in [("fixed", L_FIXED), ("scaled", max(2, W // 2))]:
            print(f"\n--- W={W}  strategy={label}  L={L}  n_realizations={N_REALIZATIONS} ---")
            stats_by_x = R.sweep_lightcone_probe_all(
                W=W, p_values=[P], L_values=[L], n_realizations=N_REALIZATIONS,
                anchor_parity_at_perturbation=ANCHOR, n_shots=1,
            )
            var, n_sites = bulk_var(stats_by_x, L)
            results[(W, label)] = (L, var, n_sites)
            print(f"W={W}  {label} (L={L}): bulk Var(d_r)={var:.4f}  "
                  f"(averaged over {n_sites} bulk site(s))  "
                  f"implied n for se={TARGET_SE}: {var / TARGET_SE**2:.0f}")

    R._save_partial({"config": dict(W_VALUES=W_VALUES, L_FIXED=L_FIXED,
                                     N_REALIZATIONS=N_REALIZATIONS, P=P),
                      "results": results}, RESULTS_PATH)

    # ── Plot ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for label, color in [("fixed", "crimson"), ("scaled", "steelblue")]:
        Ws = [W for W in W_VALUES]
        Vs = [results[(W, label)][1] for W in Ws]
        Ls = [results[(W, label)][0] for W in Ws]
        n_needed = [v / TARGET_SE**2 for v in Vs]

        lbl = f"L={L_FIXED} (fixed)" if label == "fixed" else "L=W//2 (scaled)"
        ax1.plot(Ws, Vs, "o-", color=color, label=lbl)
        ax2.plot(Ws, n_needed, "o-", color=color, label=lbl)

        # simple linear extrapolation to W_PROJECT
        coeffs = np.polyfit(Ws, Vs, 1)
        v_proj = np.polyval(coeffs, W_PROJECT)
        n_proj = v_proj / TARGET_SE**2
        print(f"\n[{label}] linear fit: Var(W) ~ {coeffs[0]:.5f}*W + {coeffs[1]:.4f}")
        print(f"[{label}] projected at W={W_PROJECT}: Var~{v_proj:.4f}  "
              f"n_realizations for se={TARGET_SE} ~ {n_proj:.0f}")
        ax2.plot([W_PROJECT], [n_proj], "*", color=color, markersize=16,
                  label=f"{lbl} projected @ W={W_PROJECT}: n~{n_proj:.0f}")

    ax1.set_xlabel("W")
    ax1.set_ylabel(r"bulk $\mathrm{Var}(d_r)$")
    ax1.set_title(f"Per-realization variance vs system size (n={N_REALIZATIONS}, p={P})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("W")
    ax2.set_ylabel(f"realizations needed for se={TARGET_SE}")
    ax2.set_title(f"Implied n_realizations vs W (extrapolated to W={W_PROJECT})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {FIGURE_PATH}")
