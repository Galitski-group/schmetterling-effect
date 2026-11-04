"""
error_analysis_lightcone_paper_w18.py
========================================
Rigorous Monte Carlo error analysis for figs/lightcone_paper_w18.png (the
W=18, noiseless, p=0.0/0.025/0.05 light-cone heatmap). Unlike open_otoc.json
(externally pre-averaged, see error_analysis_open_otoc.py), we generated
this data ourselves and the checkpoint keeps every one of the 1000 raw
per-realization d_r values at every (p, L, x) point -- so this is an EXACT
empirical standard error, not a bound.

Deliberately bypasses run_disorder_avg.py (loads the checkpoint pickle
directly): src/transfer_matrix_otoc.py is currently broken (a notebook was
accidentally saved over it, so it's raw JSON, not Python -- confirmed via
`git show 458f281:src/transfer_matrix_otoc.py`, the last commit where it
was still valid Python; even that version's ButterflyTransferMatrix-vs-
TransferMatrix naming and get_OTOC/compute_otoc interface don't match what
the rest of the codebase expects, so the actually-working version this
session used was apparently never committed). None of that module is
needed here -- this script only needs the raw checkpoint arrays and numpy.

METHOD: for each (p, L, x), n=1000 iid realizations of d_r.
    mean   = sample mean                      (the plotted C(x,t) value)
    sd     = sample std, ddof=1
    SE     = sd / sqrt(n)                      (standard error of the mean)
    CI95   = mean +/- 1.96 * SE                (normal approx, n=1000 is plenty)

OUTPUTS:
  figs/lightcone_paper_w18_errors.png  -- companion 3-panel heatmap of SE(x,t)
      (same layout/cmap convention as the original, RdPu, ggplot style)
  tmp/lightcone_paper_w18_error_summary.csv -- full per-(p,L,x) table
  Printed summary: SE range/median per p, plus a handful of representative
  points worth quoting directly in text (peak, mid-cone, late-time bulk).

USAGE:
    .butterfly/bin/python src/error_analysis_lightcone_paper_w18.py
"""

import csv
import pathlib
import pickle

import numpy as np
import matplotlib.pyplot as plt

CHECKPOINT_PATH = "partials/lightcone_paper_w18_checkpoint.pkl"
FIGURE_PATH = "figs/lightcone_paper_w18_errors.png"
CSV_PATH = "tmp/lightcone_paper_w18_error_summary.csv"
W = 18
SEED = 1962
P_VALUES = [0.0, 0.025, 0.05]  # matches the published figure; the leftover
    # p=0.01 points in the checkpoint (superseded, see conversation history)
    # are intentionally excluded.

if __name__ == "__main__":
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)

    L_values = sorted({L for (w, s, p, L) in checkpoint if w == W and s == SEED})
    x_values = sorted({x for (w, s, p, L), by_x in checkpoint.items()
                        if w == W and s == SEED for x in by_x})

    # ---- compute mean/sd/SE/CI95 for every (p, L, x) ----------------------
    rows = []  # for the CSV
    se_grid = {p: np.full((len(x_values), len(L_values)), np.nan) for p in P_VALUES}
    mean_grid = {p: np.full((len(x_values), len(L_values)), np.nan) for p in P_VALUES}
    for p in P_VALUES:
        for L in L_values:
            key = (W, SEED, p, L)
            if key not in checkpoint:
                continue
            by_x = checkpoint[key]
            for x, vals in by_x.items():
                vals = np.asarray(vals, dtype=float)
                n = len(vals)
                mean = vals.mean()
                sd = vals.std(ddof=1)
                se = sd / np.sqrt(n)
                ci95 = 1.96 * se
                rows.append(dict(p=p, L=L, x=x, n=n, mean=mean, sd=sd, se=se, ci95=ci95))
                i = x_values.index(x)
                j = L_values.index(L)
                se_grid[p][i, j] = se
                mean_grid[p][i, j] = mean

    pathlib.Path(CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "L", "x", "n", "mean", "sd", "se", "ci95"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Full per-point table ({len(rows)} rows) -> {CSV_PATH}")

    # ---- summary stats per p ------------------------------------------------
    print()
    print(f"{'p':>7} {'n':>6} {'SE min':>9} {'SE median':>10} {'SE max':>9} "
          f"{'rel.SE median (|C|>0.1)':>26}")
    for p in P_VALUES:
        ses = np.array([r["se"] for r in rows if r["p"] == p])
        means = np.array([r["mean"] for r in rows if r["p"] == p])
        rel = ses[np.abs(means) > 0.1] / np.abs(means[np.abs(means) > 0.1])
        print(f"{p:>7.3f} {1000:>6d} {ses.min():>9.4f} {np.median(ses):>10.4f} "
              f"{ses.max():>9.4f} {np.median(rel):>26.4f}")

    # ---- a few representative, directly-quotable points ----------------------
    print()
    print("Representative points (mean +/- SE, 95% CI):")
    reps = [
        ("peak, L=1, x=0", 1, 0),
        ("mid-cone, L=9, x=0", 9, 0),
        ("late bulk, L=18, x=0", 18, 0),
        ("cone edge, L=9, x=9" if 9 in x_values else None, 9, 9),
    ]
    for label, L, x in [r for r in reps if r[0] is not None]:
        for p in P_VALUES:
            match = [r for r in rows if r["p"] == p and r["L"] == L and r["x"] == x]
            if not match:
                continue
            r = match[0]
            print(f"  p={p:.3f}  {label:22s}  C = {r['mean']:+.4f} +/- {r['se']:.4f}"
                  f"   (95% CI [{r['mean']-r['ci95']:+.4f}, {r['mean']+r['ci95']:+.4f}])")

    # ---- companion SE(x,t) heatmap, same visual convention as the original --
    L_arr = np.array(L_values, dtype=float)
    x_arr = np.array(x_values, dtype=float)
    with plt.style.context("ggplot"):
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        vmax = max(np.nanmax(se_grid[p]) for p in P_VALUES)
        mesh = None
        for ax, p in zip(axes, P_VALUES):
            mesh = ax.pcolormesh(L_arr, x_arr, se_grid[p], shading="nearest",
                                  cmap="RdPu", vmin=0, vmax=vmax)
            ax.plot(L_arr, L_arr, "--", color="black", lw=1, alpha=0.6)
            ax.plot(L_arr, -L_arr, "--", color="black", lw=1, alpha=0.6)
            ax.set_xlim(L_arr.min() - 0.5, L_arr.max() + 0.5)
            ax.set_ylim(x_arr.min() - 0.5, x_arr.max() + 0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("t")
            ax.set_title(f"p = {p:.3f}")
        axes[0].set_ylabel("x")
        fig.colorbar(mesh, ax=axes, label="SE(C)")
        fig.suptitle("Standard error of C(x,t) -- n=1000 realizations/point")
        fig.savefig(FIGURE_PATH, dpi=150)
        plt.close(fig)
    print(f"\nFigure saved → {FIGURE_PATH}")
