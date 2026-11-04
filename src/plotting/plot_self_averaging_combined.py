"""
plot_self_averaging_combined.py
===================================
Combines three self-averaging figures into one 1x3 panel, (a)(b)(c):
  (a) self_averaging_scan.py's N-scan: Var[C]/|<C>|^2 vs N (fixed L/N ratio)
  (b) self_averaging_time_trace.py's L-trace: Var[C]/|<C>|^2 vs L (fixed N=16)
  (c) self_averaging_time_trace.py's L-trace: Var[C] vs L (fixed N=16)

No titles -- N=16 is folded into the legend entries of (b)/(c) instead,
since that's the only place it was otherwise communicated. L=0 (trivial,
deterministic point) is dropped from (b)/(c), same as the original
individual figures. Reads the already-computed checkpoints directly; does
NOT touch or regenerate the original individual figures
(self_averaging_scan.png, self_averaging_time_trace.png,
self_averaging_time_trace_variance.png), which are left as-is.

USAGE:
    .butterfly/bin/python src/plot_self_averaging_combined.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib.pyplot as plt
import runs.run_disorder_avg as R
import error.self_averaging_scan as S
import error.self_averaging_time_trace as T

FIGURE_PATH = str(pathlib.Path(R.FIGURE_DIR) / "self_averaging_combined.png")
PANEL_LABEL_FONTSIZE = 18

if __name__ == "__main__":
    scan_ckpt = R._load_partial(S.CHECKPOINT_PATH)
    trace_ckpt = R._load_partial(T.CHECKPOINT_PATH)

    cmap = plt.get_cmap("RdPu")

    def _colors_for(p_values):
        n = len(p_values)
        return {
            p: cmap(0.9 - 0.5 * (i / max(1, n - 1)))
            for i, p in enumerate(sorted(p_values))
        }

    with plt.style.context("ggplot"):
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

        # (a) N-scan
        colors_a = _colors_for(S.P_VALUES)
        for p in S.P_VALUES:
            Ns = [N for N in S.N_VALUES if (N, p) in scan_ckpt]
            ratios = [scan_ckpt[(N, p)]["ratio"] for N in Ns]
            axes[0].plot(Ns, ratios, "o-", color=colors_a[p], label=f"p={p:.4f}")
        axes[0].set_xlabel(r"$N$")
        axes[0].set_ylabel(r"$\mathrm{Var}[C]/|\langle C\rangle|^2$")
        axes[0].legend()

        # (b) L-trace, ratio
        colors_b = _colors_for(T.P_VALUES)
        for p in T.P_VALUES:
            Ls = [L for L in T.L_VALUES if (L, p) in trace_ckpt and L != 0]
            ratios = [trace_ckpt[(L, p)]["ratio"] for L in Ls]
            axes[1].plot(Ls, ratios, "o-", color=colors_b[p], label=f"p={p:.4f}, N={T.N}")
        axes[1].set_xlabel(r"$L$")
        axes[1].set_ylabel(r"$\mathrm{Var}[C]/|\langle C\rangle|^2$")
        axes[1].legend()

        # (c) L-trace, variance
        for p in T.P_VALUES:
            Ls = [L for L in T.L_VALUES if (L, p) in trace_ckpt and L != 0]
            vars_ = [trace_ckpt[(L, p)]["var"] for L in Ls]
            axes[2].plot(Ls, vars_, "o-", color=colors_b[p], label=f"p={p:.4f}, N={T.N}")
        axes[2].set_xlabel(r"$L$")
        axes[2].set_ylabel(r"$\mathrm{Var}[C]$")
        axes[2].legend()

        for ax, label in zip(axes, [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$", r"$\mathbf{(c)}$"]):
            ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=PANEL_LABEL_FONTSIZE,
                    va="bottom", ha="right")

        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        plt.close(fig)

    print(f"Figure saved → {FIGURE_PATH}")
