"""
lightcone_cluster_reduce.py
==============================
Master-node collation step: scans {out_dir}/{mode}/*.pkl (one file per
worker task, see lightcone_cluster_worker.py) for a given mode, merges
every (p, L) point's realization-chunks back into the standard checkpoint
format ({(W, base_seed, p, L): {x: [d_r, ...]}}) that compute_C_t /
plot_lightcone_heatmap already understand, and produces the heatmap
figure. Run once per mode, after that mode's SLURM array job completes
(e.g. via --dependency=afterok:<array_job_id>, see
slurm/lightcone_w24_reduce.slurm).

Safe to run at any point, even mid-sweep: (p, L) points with fewer than
n_realizations_total collected realizations are reported as INCOMPLETE
(missing chunks listed) and included with however many realizations they
do have — plot_lightcone_heatmap renders whatever's there, same partial-
progress convention as every other plotting script in this project.

USAGE:
    .butterfly/bin/python src/lightcone_cluster_reduce.py --mode noiseless \\
        --out-dir /path/to/shared/scratch/lightcone_w24
"""

import argparse
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import run_disorder_avg as R


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["noisy", "noiseless"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--base-seed", type=int, default=1962)
    ap.add_argument("--n-realizations-total", type=int, default=2000)
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument("--figure-path", default=None)
    args = ap.parse_args()

    checkpoint_path = args.checkpoint_path or str(
        pathlib.Path(R.PARTIALS_DIR) / f"lightcone_cluster_w{args.W}_{args.mode}_checkpoint.pkl"
    )
    figure_path = args.figure_path or str(
        R.FIGURE_DIR / f"lightcone_cluster_w{args.W}_{args.mode}.png"
    )

    mode_dir = pathlib.Path(args.out_dir) / args.mode
    files = sorted(mode_dir.glob("p*_L*_r*.pkl"))
    if not files:
        print(f"No worker output files found under {mode_dir} — nothing to reduce.")
        return

    print(f"Found {len(files)} worker output file(s) under {mode_dir}")

    # (p, L) -> x -> list of d_r values, accumulated across every chunk file
    merged: dict[tuple, dict[int, list]] = {}
    seen_ranges: dict[tuple, list[tuple[int, int]]] = {}
    for path in files:
        with open(path, "rb") as f:
            rec = pickle.load(f)
        key = (rec["p"], rec["L"])
        merged.setdefault(key, {})
        for x, vals in rec["d_r_by_x"].items():
            merged[key].setdefault(x, []).extend(vals)
        seen_ranges.setdefault(key, []).append(
            (rec["realization_start"], rec["realization_start"] + rec["realization_count"])
        )

    checkpoint: dict[tuple, dict[int, list]] = {}
    for (p, L), by_x in sorted(merged.items()):
        n = len(next(iter(by_x.values())))
        ranges = sorted(seen_ranges[(p, L)])
        covered = sum(b - a for a, b in ranges)
        status = "OK" if covered >= args.n_realizations_total else (
            f"INCOMPLETE ({covered}/{args.n_realizations_total})"
        )
        print(f"  p={p:.3f}  L={L:3d}  n={n}  chunks={len(ranges)}  {status}")
        checkpoint[(args.W, args.base_seed, p, L)] = by_x

    R._save_partial(checkpoint, checkpoint_path)

    raw_by_x: dict[int, dict[float, dict[int, np.ndarray]]] = {}
    for (w, s, p, L), by_x in checkpoint.items():
        for x, vals in by_x.items():
            raw_by_x.setdefault(x, {}).setdefault(p, {})[L] = np.array(vals)

    stats_by_x = {x: R.compute_C_t(pd) for x, pd in raw_by_x.items()}
    p_values = sorted({p for (_, _, p, _) in checkpoint})
    L_values = sorted({L for (_, _, _, L) in checkpoint})

    R.plot_lightcone_heatmap(
        stats_by_x, L_values=L_values, p_values=p_values, stat="C",
        figure_path=figure_path, cmap="RdPu", cone_line_color="black",
    )


if __name__ == "__main__":
    main()
