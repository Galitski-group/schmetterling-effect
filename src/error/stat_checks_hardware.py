"""
stat_checks_hardware.py
==========================
Null tests and theory-agreement tests for the W=8 hardware OTOC data.
Reads checkpoints directly and has its own transfer-matrix code, so it does
not depend on run_disorder_avg.py or src/transfer_matrix_otoc.py.

Transfer-matrix conventions compared:
  plotted : seed pair (pert, pert+1), then L layers starting odd, read at t=L.
            This is what plot_h2_2_results.py draws.
  circuit : seed on the pair the circuit's first backward layer actually
            uses, then L-1 more layers. Same number of gate layers as the
            circuit, and T(p) on exactly the layers that can be measured.

Errors come from bootstrapping realizations (hardware: whole circuits), so
correlations between sites and within a tiled circuit are carried through.

USAGE:
    .butterfly/bin/python src/stat_checks_hardware.py
"""

import csv
import pathlib
import pickle

import numpy as np
from scipy import stats

N_BOOT = 4000
RNG_SEED = 20261103
OUT_DIR = pathlib.Path("tmp/stat_checks")


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def transfer_matrix(p):
    a = (6 + 2 * p - p**2) / 30
    b = 4 * (3 - 4 * p + 2 * p**2) / 15
    c = p * (8 - 8 * p + 4 * p**2 - p**3) / 60
    d = (3 + 8 * (p - 1) ** 2 + 4 * (p - 1) ** 4) / 15
    return np.array([[1, a, a, c], [0, 0, 0, 0], [0, 0, 0, 0], [0, b, b, d]])


def pairs(W, parity):
    return [(i, i + 1) for i in range(parity, W - 1, 2)]


def apply_layer(M, W, parity, T):
    for i, j in pairs(W, parity):
        axes = [i, j] + [k for k in range(W) if k not in (i, j)]
        perm = np.transpose(M, axes)
        out = (T @ perm.reshape(4, -1)).reshape([2, 2] + list(perm.shape[2:]))
        M = np.transpose(out, np.argsort(axes))
    return M


def profile(M, W):
    """C/2 at every site, the normalization the plotted comparison uses."""
    return np.array([
        M.sum(axis=tuple(k for k in range(W) if k != q))[1] / 2 for q in range(W)
    ])


def evolve(W, pert, parities, p):
    """Seed on the first layer's pair containing pert, then apply the rest."""
    T = transfer_matrix(p)
    pair = next(ab for ab in pairs(W, parities[0]) if pert in ab)
    idx = [0] * W
    idx[pair[0]] = idx[pair[1]] = 1
    M = np.zeros((2,) * W)
    M[tuple(idx)] = 32 / 15
    for parity in parities[1:]:
        M = apply_layer(M, W, parity, T)
    return profile(M, W)


def tm_circuit_anchored_odd(W, pert, L, p):
    return evolve(W, pert, [(k + 1) % 2 for k in range(L)], p)


def tm_circuit_unanchored(W, pert, L, p):
    return evolve(W, pert, [(L - 1 - k) % 2 for k in range(L)], p)


def tm_plotted(W, pert, L, p):
    return evolve(W, pert, [0] + [t % 2 for t in range(1, L + 1)], p)


def clusters(n, sizes=None):
    """Index arrays to resample: whole circuits if sizes given, else singles."""
    if sizes is None:
        return [np.array([i]) for i in range(n)]
    out, i = [], 0
    for s in sizes:
        out.append(np.arange(i, min(i + s, n)))
        i += s
    return out


def boot_means(values, groups, rng, n_boot=N_BOOT):
    """Bootstrap means of a (n_realizations, n_cells) array."""
    g = len(groups)
    sums = np.array([values[ix].sum(axis=0) for ix in groups])
    counts = np.array([len(ix) for ix in groups])
    picks = rng.integers(0, g, size=(n_boot, g))
    return sums[picks].sum(axis=1) / counts[picks].sum(axis=1)[:, None]


def gof(blocks, rng):
    """Chi-square-like distance from theory, calibrated by bootstrap.

    blocks: list of (values, theory, groups), one per independent L.
    Cells with no observed variance use a floor of one event, 2/n.
    Returns statistic, cell count, p-value, mean residual and its SE.
    """
    stat, cells, null, resid, resid_boot = 0.0, 0, np.zeros(N_BOOT), [], []
    for values, theory, groups in blocks:
        n = values.shape[0]
        mean = values.mean(axis=0)
        boot = boot_means(values, groups, rng)
        se = np.maximum(boot.std(axis=0, ddof=1), 2 / n)
        stat += np.sum(((mean - theory) / se) ** 2)
        null += np.sum(((boot - mean) / se) ** 2, axis=1)
        cells += len(theory)
        resid.append(mean - theory)
        resid_boot.append(boot - theory)
    p_value = (1 + np.sum(null >= stat)) / (N_BOOT + 1)
    resid = np.concatenate(resid)
    resid_boot = np.concatenate(resid_boot, axis=1).mean(axis=1)
    return stat, cells, p_value, resid.mean(), resid_boot.std(ddof=1)


def attenuation(blocks, rng):
    """Least-squares alpha in data = alpha * theory, with a bootstrap CI."""
    num, den, num_boot = 0.0, 0.0, np.zeros(N_BOOT)
    for values, theory, groups in blocks:
        num += np.sum(values.mean(axis=0) * theory)
        den += np.sum(theory**2)
        num_boot += np.sum(boot_means(values, groups, rng) * theory, axis=1)
    alpha_boot = num_boot / den
    lo, hi = np.percentile(alpha_boot, [2.5, 97.5])
    return num / den, lo, hi


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.zeros(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def matrix(by_x, W):
    pert = W // 2
    return np.array([by_x[q - pert] for q in range(W)], dtype=float).T


def main():
    rng = np.random.default_rng(RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hw = load("partials/lightcone_h2_2_tiled_checkpoint.pkl")
    hw_raw = load("partials/lightcone_h2_2_tiled_raw_results.pkl")
    noiseless = load("partials/lightcone_w8_noiseless_checkpoint.pkl")
    noisy = load("partials/lightcone_w8_noisy_checkpoint.pkl")
    w18 = load("partials/lightcone_paper_w18_checkpoint.pkl")
    single = load("partials/h2_2_tiled_checkpoint.pkl")
    single_raw = load("partials/h2_2_tiled_raw_results.pkl")

    W, pert, seed = 8, 4, 1962
    ps = [0.0, 0.025, 0.05]
    Ls = list(range(1, 9))

    def sizes(L, p):
        return [n for _, n in hw_raw[(W, seed, p, L)]["chunk_meta"]["pert"]]

    theory = {(p, L): tm_circuit_anchored_odd(W, pert, L, p) for p in ps for L in Ls}

    # ---- 1. Is the theory right? Noiseless simulation should match exactly.
    print("=" * 78)
    print("1. THEORY VALIDATION against noiseless simulation (no hardware)")
    print("=" * 78)
    print(f"{'data':>10} {'p':>6} {'cells':>6} {'chi2':>9} {'chi2/cell':>10} "
          f"{'p-value':>9} {'mean resid':>11} {'SE':>8}")
    validation_rows = []
    for p in ps:
        blocks = [(matrix(noiseless[(W, seed, p, L)], W), theory[(p, L)],
                   clusters(180)) for L in Ls]
        validation_rows.append(("W=8", p, gof(blocks, rng)))
    for p in [0.0, 0.01, 0.025, 0.05]:
        M, blocks = None, []
        T = transfer_matrix(p)
        for L in range(1, 19):
            if L == 1:
                pair = next(ab for ab in pairs(18, 1) if 9 in ab)
                idx = [0] * 18
                idx[pair[0]] = idx[pair[1]] = 1
                M = np.zeros((2,) * 18)
                M[tuple(idx)] = 32 / 15
            else:
                M = apply_layer(M, 18, L % 2, T)
            blocks.append((matrix(w18[(18, seed, p, L)], 18), profile(M, 18),
                           clusters(1000)))
        validation_rows.append(("W=18", p, gof(blocks, rng)))
    for name, p, (stat, cells, pv, res, res_se) in validation_rows:
        print(f"{name:>10} {p:>6} {cells:>6} {stat:>9.1f} {stat/cells:>10.3f} "
              f"{pv:>9.4f} {res:>+11.4f} {res_se:>8.4f}")

    # ---- 2. Null: L=1 has no measurements, so every p is the same experiment.
    print()
    print("=" * 78)
    print("2. NULL TEST: at L=1 the three p values are the same circuits")
    print("=" * 78)
    print(f"{'data':>10} {'pair':>16} {'x':>3} {'difference':>11} {'SE':>8} {'z':>7}")
    null_p = []
    for name, ck, use_clusters in [("hardware", hw, True), ("noisy sim", noisy, False)]:
        for a, b in [(0.0, 0.025), (0.0, 0.05), (0.025, 0.05)]:
            A, B = matrix(ck[(W, seed, a, 1)], W), matrix(ck[(W, seed, b, 1)], W)
            for x in (-1, 0):
                q = x + pert
                # different jobs, so uncertainties add; circuits are resampled per job
                ga = clusters(180, sizes(1, a)) if use_clusters else clusters(180)
                gb = clusters(180, sizes(1, b)) if use_clusters else clusters(180)
                se = np.hypot(boot_means(A[:, [q]], ga, rng).std(ddof=1),
                              boot_means(B[:, [q]], gb, rng).std(ddof=1))
                diff = A[:, q].mean() - B[:, q].mean()
                z = diff / se
                null_p.append((name, a, b, x, diff, se, z, 2 * stats.norm.sf(abs(z))))
                print(f"{name:>10} {f'{a} vs {b}':>16} {x:>3} {diff:>+11.3f} {se:>8.3f} {z:>+7.2f}")
    hw_p = np.array([r[7] for r in null_p if r[0] == "hardware"])
    print(f"hardware: smallest raw p = {hw_p.min():.3f}, after Holm = {holm(hw_p).min():.3f}")

    # ---- 3. Null: outside the cone the signal must be exactly zero on average.
    print()
    print("=" * 78)
    print("3. NULL TEST: outside the causal cone, C should average to 0")
    print("=" * 78)
    print(f"{'data':>10} {'p':>6} {'cells':>6} {'mean C':>9} {'SE':>8} {'z':>7} "
          f"{'nonzero shots':>14} {'rate':>8}")
    outside_rows = []
    for name, ck, use_clusters in [("hardware", hw, True), ("noisy sim", noisy, False),
                                    ("noiseless", noiseless, False)]:
        for p in ps:
            total, cells, boot_total, nonzero, shots = 0.0, 0, np.zeros(N_BOOT), 0, 0
            for L in Ls:
                out = np.where(theory[(p, L)] == 0)[0]
                if len(out) == 0:
                    continue
                vals = matrix(ck[(W, seed, p, L)], W)[:, out]
                g = clusters(180, sizes(L, p)) if use_clusters else clusters(180)
                total += vals.mean(axis=0).sum()
                boot_total += boot_means(vals, g, rng).sum(axis=1)
                cells += len(out)
                nonzero += int(np.count_nonzero(vals))
                shots += vals.size
            mean, se = total / cells, boot_total.std(ddof=1) / cells
            z = mean / se if se > 0 else 0.0
            outside_rows.append((name, p, mean, se, z, nonzero / shots))
            print(f"{name:>10} {p:>6} {cells:>6} {mean:>+9.4f} {se:>8.4f} {z:>+7.2f} "
                  f"{f'{nonzero}/{shots}':>14} {nonzero/shots:>8.4f}")

    # ---- 4. Hardware against theory and against the noise model.
    print()
    print("=" * 78)
    print("4. FRONT DATA: hardware vs theory and vs noisy simulation")
    print("=" * 78)
    print(f"{'comparison':>24} {'p':>6} {'cells':>6} {'chi2/cell':>10} {'p-value':>9} "
          f"{'mean resid':>11} {'SE':>8}")
    comparisons = []
    for p in ps:
        hw_blocks = [(matrix(hw[(W, seed, p, L)], W), theory[(p, L)],
                      clusters(180, sizes(L, p))) for L in Ls]
        comparisons.append(("hardware - theory", p, gof(hw_blocks, rng)))
        noisy_blocks = [(matrix(noisy[(W, seed, p, L)], W), theory[(p, L)],
                         clusters(180)) for L in Ls]
        comparisons.append(("noisy sim - theory", p, gof(noisy_blocks, rng)))
        diff_blocks = [(matrix(hw[(W, seed, p, L)], W) - matrix(noisy[(W, seed, p, L)], W),
                        np.zeros(W), clusters(180, sizes(L, p))) for L in Ls]
        comparisons.append(("hardware - noisy sim", p, gof(diff_blocks, rng)))
    adj = holm(np.array([c[2][2] for c in comparisons]))
    for (name, p, (stat, cells, pv, res, res_se)), a in zip(comparisons, adj):
        print(f"{name:>24} {p:>6} {cells:>6} {stat/cells:>10.3f} {pv:>9.4f} "
              f"{res:>+11.4f} {res_se:>8.4f}   Holm p={a:.4f}")

    print()
    print("Signal retained, data = alpha * theory (95% bootstrap CI):")
    print(f"{'p':>6} {'L range':>8} {'hardware alpha':>26} {'noisy sim alpha':>26}")
    alpha_rows = []
    for p in ps:
        for label, rng_L in [("1-4", range(1, 5)), ("5-8", range(5, 9)), ("all", range(1, 9))]:
            h = [(matrix(hw[(W, seed, p, L)], W), theory[(p, L)],
                  clusters(180, sizes(L, p))) for L in rng_L]
            n = [(matrix(noisy[(W, seed, p, L)], W), theory[(p, L)],
                  clusters(180)) for L in rng_L]
            ah, an = attenuation(h, rng), attenuation(n, rng)
            alpha_rows.append((p, label, *ah, *an))
            print(f"{p:>6} {label:>8} {f'{ah[0]:.3f} [{ah[1]:.3f}, {ah[2]:.3f}]':>26} "
                  f"{f'{an[0]:.3f} [{an[1]:.3f}, {an[2]:.3f}]':>26}")

    # ---- 5. Single-probe hardware points against both theory conventions.
    print()
    print("=" * 78)
    print("5. SINGLE-QUBIT DATA: hardware vs both transfer-matrix conventions")
    print("=" * 78)
    print(f"{'p':>6} {'L':>3} {'C':>8} {'SE':>7} {'SE circ':>8} {'plotted':>8} {'z':>7} "
          f"{'circuit':>8} {'z':>7}")
    rows, chi = [], {}
    for (w, s, p, L) in sorted(single, key=lambda k: (k[2], k[3])):
        v = np.asarray(single[(w, s, p, L)], dtype=float)
        n = len(v)
        chunk = [c for _, c in single_raw.get((w, s, p, L), {}).get("chunk_meta", {}).get("pert", [])]
        se = v.std(ddof=1) / np.sqrt(n)
        if chunk:
            se_c = boot_means(v[:, None], clusters(n, chunk), rng).std(ddof=1)
        else:
            se_c = float("nan")
        se_use = np.nanmax([se, se_c])
        tp, tc = tm_plotted(W, pert, L, p)[0], tm_circuit_unanchored(W, pert, L, p)[0]
        zp, zc = (v.mean() - tp) / se_use, (v.mean() - tc) / se_use
        chi.setdefault(p, [0.0, 0.0, 0])
        chi[p][0] += zp**2
        chi[p][1] += zc**2
        chi[p][2] += 1
        rows.append((p, L, v.mean(), se, se_c, tp, zp, tc, zc))
        print(f"{p:>6} {L:>3} {v.mean():>+8.4f} {se:>7.4f} {se_c:>8.4f} {tp:>8.4f} {zp:>+7.2f} "
              f"{tc:>8.4f} {zc:>+7.2f}")
    print()
    print("Chi-square over all L (independent jobs, larger of the two SEs):")
    for p, (cp, cc, k) in chi.items():
        print(f"  p={p}: plotted curve chi2={cp:.1f}/{k} (p={stats.chi2.sf(cp, k):.2g}), "
              f"circuit-matched chi2={cc:.1f}/{k} (p={stats.chi2.sf(cc, k):.2g})")

    with open(OUT_DIR / "single_qubit_vs_theory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "L", "C", "se", "se_circuit", "tm_plotted", "z_plotted",
                         "tm_circuit", "z_circuit"])
        writer.writerows(rows)
    with open(OUT_DIR / "front_vs_theory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "L", "x", "theory", "hardware", "se_circuit", "z",
                         "noisy_sim", "noiseless_sim"])
        for p in ps:
            for L in Ls:
                H = matrix(hw[(W, seed, p, L)], W)
                se = boot_means(H, clusters(180, sizes(L, p)), rng).std(axis=0, ddof=1)
                N, Q = matrix(noisy[(W, seed, p, L)], W), matrix(noiseless[(W, seed, p, L)], W)
                for q in range(W):
                    th, m = theory[(p, L)][q], H[:, q].mean()
                    z = (m - th) / max(se[q], 2 / 180)
                    writer.writerow([p, L, q - pert, th, m, se[q], z,
                                     N[:, q].mean(), Q[:, q].mean()])
    print(f"\nTables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
