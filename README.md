# Schmetterling Effect

*Schmetterling* is German for butterfly. This project measures how a small
disturbance spreads through a quantum computer, the quantum version of the
butterfly effect, and what happens to that spreading when qubits are also being
measured along the way.

We do this three ways and compare them:

1. **Real hardware:** Quantinuum H-series trapped-ion devices.
2. **Circuit simulation:** the same circuits on a classical simulator, with or
   without a noise model.
3. **Theory:** a classical calculation, the *transfer matrix*, that gives the
   average answer without simulating any quantum states.

This README assumes you know what a qubit and a quantum gate are. Everything
else is introduced as it comes up.

> **Repository status (November 2026).** Scripts now live in subfolders by purpose
> (see [How the repository is organized](#how-the-repository-is-organized)), but
> many internal paths still point to the old locations. Read
> [Current state of the code](#current-state-of-the-code) before running anything
> other than the quickstart.

---

## The physics in five steps

### 1. The butterfly effect needs a quantum definition

In classical chaos, you nudge the initial condition and watch two trajectories
separate. Quantum mechanics has no trajectories to compare, so we need another
way to ask whether a small local change has affected a distant part of the system.

The answer is an **echo**. Run the dynamics forward, disturb one qubit, then run
the dynamics exactly backward:

```text
prepare state  →  U  →  disturb one qubit  →  U†  →  read out another qubit
```

- **Without the disturbance**, `U†` undoes `U` and the state returns to where it
  started.
- **With the disturbance**, the undoing is imperfect. Whether a distant qubit
  notices depends on whether information from the disturbed site has reached it.

The disturbed qubit is the **perturbation site** and the qubit we read is the
**probe**. This construction is an **out-of-time-order correlator (OTOC)**. The
name refers to its operators not being in time order: the circuit evolves
forward, then backward.

### 2. What we measure

We run each circuit twice, with and without the disturbance, and read the probe
as `+1` or `-1`. The signal is the difference:

```text
d = (probe outcome without disturbance) − (probe outcome with disturbance)

C(x, t) = average of d
```

Here `x` is the probe's distance from the perturbation and `t` is the circuit
depth, the number of layers `L` in `U`. A single `d` can only be `-2`, `0` or `+2`.

Two limits make the numbers meaningful:

- **`C ≈ 0`:** the disturbance has not reached the probe.
- **`C ≈ 1`:** the disturbance has fully scrambled the probe. The undisturbed echo
  returns the probe to `+1`, while the disturbed run gives `+1` and `-1` equally
  often, so its average is 0. The difference is `1 − 0 = 1`.

### 3. The circuit: a random brick wall

`U` is built from layers of two-qubit gates arranged like bricks, with each layer
offset by one site:

```text
layer 0:  [0 1] [2 3] [4 5] [6 7]
layer 1:     [1 2] [3 4] [5 6]
layer 2:  [0 1] [2 3] [4 5] [6 7]
```

Each gate is an independent **Haar-random** SU(4) gate, meaning it is drawn
uniformly from all possible two-qubit unitaries. The randomness gives us generic
chaotic dynamics with no special structure.

One draw of all the random choices is a **disorder realization**. We care about
the average over many realizations, and we often take just one shot per
realization, since a new circuit gives more independent information than
repeating an old one.

Gates act only on neighbors, so information can move at most one site per layer.
That limit defines a **light cone**: outside it, `C` must be exactly zero on an
ideal device. Inside it, the signal front travels more slowly. For qubits it moves
at about 3/5 of the maximum speed and broadens as it goes
([Nahum, Vijay & Haah 2018](https://doi.org/10.1103/PhysRevX.8.021014);
[von Keyserlingk et al. 2018](https://doi.org/10.1103/PhysRevX.8.021013)).

### 4. Adding measurements

Now the twist. With probability `p`, each eligible qubit in each layer is also
measured, and the outcome is thrown away. `U` and `U†` draw their measurement
locations independently.

A measurement can't be undone, so the echo no longer returns perfectly, even with
no disturbance. This changes the behavior:

- **`p = 0`:** `C` rises to about 1 and stays there.
- **`p > 0`:** `C` rises as the front arrives and then decays toward 0 as
  measurements destroy the signal.

### 5. Theory without simulating quantum states

Averaging over Haar-random gates turns the quantum problem into a classical one.
Each site becomes a two-state variable, empty (`0`) or occupied (`1`), and each
gate updates a pair of sites by a fixed rule:

```text
00 → 00
01 or 10 → a·00 + b·11
11 → c·00 + d·11
```

At `p = 0` the coefficients are `a = 1/5`, `b = 4/5`, `c = 0`, `d = 1`. This reads
like a game of chance: where an occupied site meets an empty one, both become
occupied with probability 4/5 and both empty with probability 1/5. That biased
step is where the front speed comes from: `4/5 − 1/5 = 3/5`.

For `p > 0`, with independent measurement sites in `U` and `U†`:

```text
a = (6 + 2p − p²) / 30
b = 4(3 − 4p + 2p²) / 15
c = p(8 − 8p + 4p² − p³) / 60
d = (3 + 8(p−1)² + 4(p−1)⁴) / 15
```

Now `a + b < 1` and `c + d < 1`, so the total weight shrinks and `C` decays.

Applying this rule layer by layer gives the **exact average** `C(x, t)`, with no
sampling noise. For very large systems, `open_brickwall_otoc.py` samples the same
process by Monte Carlo instead, which is how we reach N = 2000.

Two normalizations are easy to trip over:

- The seed weight is `2q⁴/(q⁴−1)`, which is `32/15` for qubits.
- Theory curves are divided by 2 to match the `C ≈ 1` convention above.

---

## Try it: quickstart

This builds real circuits and simulates them locally. It needs no hardware account
and takes about 25 seconds. Run it from the repository root:

```python
import importlib.util
import numpy as np
from pytket.extensions.qiskit import AerBackend

spec = importlib.util.spec_from_file_location("schmetterling", "src/__main__.py")
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

W, L, p, n = 6, 4, 0.0, 300      # qubits, depth, measurement rate, realizations
backend = AerBackend()
d = []
for r in range(n):
    shared = dict(L=L, W=W, pert_site=3, pert_op="X", probe_site=0, probe_angle=0.0,
                  p=p, seed=r, init_seed=10_000 + r, meas_seed=20_000 + r,
                  meas_seed_ud=30_000 + r, add_barrier=False)
    # Same realization, with and without the disturbance
    circuits = [sm.build_brickwall_otoc_circuit(unperturbed=True, **shared),
                sm.build_brickwall_otoc_circuit(unperturbed=False, **shared)]
    compiled = backend.get_compiled_circuits(circuits, optimisation_level=0)
    handles = backend.process_circuits(compiled, n_shots=1)
    bits = [backend.get_result(h).get_shots()[0, -1] for h in handles]
    d.append((1 - 2 * int(bits[0])) - (1 - 2 * int(bits[1])))    # bit 0/1 → +1/-1

d = np.array(d, dtype=float)
print(f"C = {d.mean():.3f} +/- {d.std(ddof=1) / np.sqrt(n):.3f}")
```

Expect about `C = 0.53 +/- 0.05`; the exact value changes slightly between runs. Try
changing `L`, or setting `p = 0.05`.

### Setup

```bash
make butterfly                  # creates the .butterfly virtual environment
source .butterfly/bin/activate
```

Python 3.10 or newer is required. This installs the project in editable mode, which
puts `src/` on Python's import path; scripts that use `import runs.…` rely on that.

Run every command from the repository root, because several scripts use paths like
`partials/…` relative to where you launch them.

---

## How a run works

### Two kinds of circuit

| Circuit | Reads out | Gives you | Builder in `src/__main__.py` |
|---|---|---|---|
| Single probe | one chosen qubit | `C(t)` at one distance | `build_brickwall_otoc_circuit` |
| Probe-all | every qubit | the full light cone `C(x, t)` | `build_brickwall_lightcone_probe_all_circuit` |

There are also `bowtie` variants, a different gate layout.

**Anchoring:** by default, gate-layer parity is counted from the start of the
circuit, so the layer touching the perturbation changes with `L`. That makes the
light cone lopsided. `anchor_parity_at_perturbation="odd"` counts from the
perturbation instead, and the light-cone runs use it.

### Sources of randomness

Each has its own seed, so any one can be varied or reproduced alone:

| Randomness | Seed |
|---|---|
| Gates | `BASE_SEED × 10 000 + c_idx` |
| Initial Pauli on the non-probe qubits | `BASE_SEED × 10 000 000 + c_idx × 1 000 + i_idx` |
| Measurement sites in `U` | `(MEAS_SEED+1) × 10 000 000 + c_idx × 1 000 + i_idx` |
| Measurement sites in `U†` | `(MEAS_SEED+2) × 10 000 000 + c_idx × 1 000 + i_idx` |

The seeds above belong to the realization-based sweeps; some scripts define their
own seeding, so check the script you are running.

### Where circuits run

| | Local simulator | Quantinuum device |
|---|---|---|
| **Selected by** | default, no device name | an explicit `--device-name` or `device_name=` |
| **Noise** | none, or a model set by `NOISE_*` | the device's own |
| **Account needed** | no | yes, Quantinuum Nexus |
| **Cost** | your CPU time | HQCs (hardware credits) |

**Hardware is opt-in.** The `DEVICE_NAME` constant in `run_disorder_avg.py` is only
a reference value, and backends default to local simulation unless you pass a
device name.

The noise model approximates H2 with gate errors of `7.3e-5` for one qubit and
`1.29e-3` for two, and a readout error of `1.35e-3`. It is a simplified model, so
agreement or disagreement with hardware is itself a result.

### Tiling: making hardware affordable

Running one circuit per realization on hardware is slow and costly. **Tiling**
packs many independent realizations into one larger circuit:

- **In space:** side-by-side copies on separate qubits.
- **In time:** reset the qubits and run further realizations afterward.

The batch size is limited by the qubit budget, the sequential depth, the classical
register (64 bits on H2-2), and the compiler's time budget. Probe-all circuits need
`W` classical bits per realization, so they hit the register limit much sooner.

Tiling has a statistical consequence: realizations in one circuit share a device
run, so they may not be fully independent. `src/error/stat_checks_hardware.py`
accounts for this by bootstrapping whole circuits.

### Long hardware queues

Jobs can sit queued for days, so don't wait on them one at a time:

```text
src/submit_lightcone_h2_2_tiled_pending.py    submit everything, then exit
src/harvest_lightcone_h2_2_tiled_pending.py   collect finished jobs; safe to rerun
src/recover_h2_2_job.py                       rescue one job if its local process died
```

### From outcomes to error bars

For each point, `compute_C_t` in `src/__main__.py` gives the mean of `d` across
realizations and its standard error, `sd / √n`. That error includes both disorder
and shot noise. Results are saved in `partials/` as checkpoints, so interrupted
runs can resume and figures can be remade without new runs.

---

## How the repository is organized

The folders follow the path a result takes, from raw circuits to a figure with
error bars:

```text
src/__main__.py          circuits, execution, statistics
        │
        ▼
src/runs/  src/zaratan/  run experiments, on a laptop, hardware or cluster
        │
        ▼
partials/                saved outcomes and checkpoints
        │
        ├──────────────▶ src/plotting/  →  figs/
        └──────────────▶ src/error/     →  uncertainties and checks
```

Separating data from figures means you can restyle a plot without rerunning
anything, which matters when the data took days on a quantum computer.

### `src/`: the core and the top-level experiments

| File | What it does |
|---|---|
| `__main__.py` | The core library: circuit builders, `_run_circuits`, `compute_C_t` |
| `open_brickwall_otoc.py` | Theory class, plus the Monte Carlo run for N = 2000 |
| `open_brickwall_otoc_exact_tm_run.py` | The same class, plus the exact N = 8 run |
| `tm_eigvals.py` | Transfer-matrix eigenvalues versus `p` |
| `transfer_matrix_otoc.py` | Older theory module (see [Current state](#current-state-of-the-code)) |
| `lightcone_paper_w18.py` | W=18 noiseless light cone for the paper |
| `lightcone_w8_noisy_vs_noiseless.py` | Simulator companion to the W=8 hardware run |
| `front_width_scaling.py` | Tests whether the front's width follows a KPZ-like power law |
| `submit_…`, `harvest_…`, `recover_…` | Hardware queue tools |
| `probe_compiled_gate_counts.py` | Gate counts after compilation |

### `src/runs/`: scripts that produce data

Start with `run_disorder_avg.py`. It is the main orchestrator, and most other
scripts import it for its sweeps, noise settings and paths.

| Scripts | Purpose |
|---|---|
| `run_h2_hardware_test.py` | Smallest end-to-end hardware test |
| `run_h2_2_tiled.py` | Tiled single-site OTOC on H2-2 |
| `run_lightcone_h2_2_tiled.py` | Tiled light cone on H2-2 |
| `run_lightcone_local.py`, `run_lightcone_probe_all_local.py` | Small local light cones |
| `run_lightcone_probe_all_w8_hw_*` | Local rehearsals at the hardware size; `anchored` and `anchored_odd` fix the layer parity at the perturbation |
| `run_lightcone_probe_all_w8_tiled_local*` | Tests the tiling code locally before using hardware |
| `run_lightcone_probe_all_w18_*` | W=18 comparisons; `p05` adds `p = 0.05` |
| `run_variance_scaling_test.py` | Projects how many realizations a larger system would need |

Most of these come in `_noisy` and `_noiseless` pairs, which differ only in whether
the noise model is on.

### `src/plotting/`: data to figures

These read saved data and run no circuits.

| Script | Figure |
|---|---|
| `plot_h2_2_vs_tm_simulation.py` | Single-site hardware data against the current theory curves |
| `plot_h2_2_results.py`, `plot_h2_2_vs_tm_styles.py`, `plot_h2_2_normalized.py` | Earlier versions of that comparison |
| `plot_lightcone_h2_2_tiled.py`, `plot_lightcone_paper_w18.py` | Light-cone heatmaps |
| `plot_open_otoc.py` | N = 2000 OTOC and Gaussian limit |
| `plot_eigvals.py` | Eigenvalues versus `p` |
| `plot_self_averaging_combined.py` | Three-panel self-averaging figure |

### `src/error/`: how much to trust the results

| Script | What it answers |
|---|---|
| `stat_checks_hardware.py` | Do the null tests pass, and does the data agree with theory? |
| `error_analysis_lightcone_paper_w18.py` | Standard errors for the W=18 light cone |
| `error_analysis_open_otoc.py` | A conditional error bound for the N = 2000 data |
| `self_averaging_scan.py` | How do relative fluctuations change with system size? |
| `self_averaging_time_trace.py` | How do they change with depth at N = 16? |

### `src/zaratan/` and `slurm/`: large sweeps on a cluster

`src/zaratan/` holds the Python for the Zaratan cluster, and `slurm/` holds the job
scripts that launch it. See [Running larger sweeps](#running-larger-sweeps-on-a-cluster).

### Everything else

| Folder | Contents | In git? |
|---|---|---|
| `partials/` | Saved data and checkpoints | No |
| `figs/` | Where scripts save figures | No |
| `figs/paper_figs/` | Figures selected for the paper | Yes |
| `figs/error_figs/` | Empty so far; the error scripts still save to `figs/` | Empty |
| `tmp/` | Logs and temporary tables | No |
| `notebooks/` | Exploratory notebooks | Only `calculate_nuotoc` and `run_experiment` |
| `tests/` | Test suite | Yes |
| `mnist_data/` | Dataset left over from the QNN work, which has moved out of this repo | Yes |

`partials/` is not in git, so a fresh clone has no data. Plotting and analysis
scripts need a copy of it.

### Which script makes which result

Paths are relative to `src/`.

| Result | Generate the data | Make the figure |
|---|---|---|
| Single-site OTOC on hardware | `runs/run_h2_2_tiled.py` | `plotting/plot_h2_2_vs_tm_simulation.py` |
| Theory curves for that figure | `open_brickwall_otoc_exact_tm_run.py` | same figure |
| Light cone on hardware | `runs/run_lightcone_h2_2_tiled.py` or submit/harvest | `plotting/plot_lightcone_h2_2_tiled.py` |
| W=8 light cone in simulation | `lightcone_w8_noisy_vs_noiseless.py` | same script |
| W=18 light cone | `lightcone_paper_w18.py` | `plotting/plot_lightcone_paper_w18.py` |
| N=2000 OTOC and Gaussian limit | `open_brickwall_otoc.py` | `plotting/plot_open_otoc.py` |
| Eigenvalues versus p | `tm_eigvals.py` | `plotting/plot_eigvals.py` |
| Self-averaging | `error/self_averaging_scan.py`, `error/self_averaging_time_trace.py` | `plotting/plot_self_averaging_combined.py` |

**Before you run the theory scripts:** `open_brickwall_otoc.py`,
`open_brickwall_otoc_exact_tm_run.py` and `tm_eigvals.py` start their calculation
as soon as they are imported, so don't import them as libraries. They also save
their JSON to `~/Desktop`. Move that file into `partials/` for the plotting scripts.

### Theory options worth knowing

`corr_sites` says whether `U` and `U†` measure the same sites. Independent sites
(`False`) are equivalent to correlated ones with `p → p(2−p)`. Our circuits draw
them independently, while the N = 2000 run uses `corr_sites=True`, so keep this in
mind when comparing figures.

The Gaussian reference curve in the N = 2000 figure is `2·exp(−κ p t²)`, with
`κ = (q²−1)(q−1) / (q(q²+1))`, which is `3/10` for qubits.

### Running larger sweeps on a cluster

For large sweeps, each Slurm task handles one chunk of realizations at one
`(p, L)` point and writes its own file, so tasks can run in any order and fail
independently:

```text
plan (task table)  →  array of workers  →  reduce (merge into a checkpoint + figure)
```

These steps are `src/zaratan/lightcone_cluster_plan.py`,
`lightcone_cluster_worker.py` and `lightcone_cluster_reduce.py`, launched by
`slurm/submit_all.sh`. Reduce can run mid-sweep and reports what is missing. The
Slurm files currently call the old script paths; see below.

---

## What we have checked, and what remains open

These come from `src/error/stat_checks_hardware.py`, which writes its tables to
`tmp/stat_checks/`. Multiple comparisons are Holm-corrected where noted in the script.

**Passing**

- **Null test at `L = 1`:** these circuits contain no mid-circuit measurements, so
  all `p` values should agree, and hardware does within error.
- **Null test outside the light cone:** hardware `C` is consistent with zero.
- **Theory at `p = 0`:** the transfer matrix matches noiseless simulation at W=18
  when its layers are aligned with the circuit (χ²/cell = 0.83).
- **Noise model at `p = 0`:** it accounts for the hardware's light-cone signal loss.

**Open**

- **Theory at `p > 0`:** noiseless simulation decays more slowly than the transfer
  matrix, and the gap grows with depth. The cause is unknown.
- **Layer counting:** the single-site theory curve appears to count one more gate
  layer than the circuit applies, which shows up clearly against direct simulation
  at `p = 0, L = 4` (0.537 ± 0.023 versus 0.437).
- **Hardware at `p > 0`:** it falls below the noise model, which suggests the model
  leaves out something that matters when measurements are present.
- **Hardware at `p = 0`, large `L`:** on the single-site data, it drops below
  theory at `L = 12` and `L = 16`.

**Apparent agreement between hardware and theory at `p = 0.05` should not be taken
as validation.** The theory has not been shown to describe this circuit at `p > 0`,
and the hardware error bars are large.

**Known limits on error bars**

- `partials/open_otoc.json` stores only means, so
  `src/error/error_analysis_open_otoc.py` gives only a conditional bound, not a true
  standard error.
- The self-averaging checkpoints store summaries, so their error bars need reruns.

---

## Current state of the code

Checked in November 2026, after the move into subfolders. Scripts marked as
blocked fail on import. The others in the broken rows import fine and fail, or write
to the wrong place, when run.

| Status | What | Why |
|---|---|---|
| Works | Quickstart and `src/__main__.py` | no dependence on moved files |
| Works | `src/error/stat_checks_hardware.py` | reads `partials/` directly, relative to the repo root |
| Blocked | `src/runs/run_disorder_avg.py` and everything importing it | looks for `src/runs/__main__.py`, one folder too deep |
| Blocked | `src/runs/run_h2_hardware_test.py` | same wrong path to `__main__.py` |
| Broken | `src/transfer_matrix_otoc.py` | contains notebook JSON rather than Python; `run_disorder_avg.py` also looks for it in `src/runs/` |
| Broken | `plot_open_otoc.py`, `plot_eigvals.py`, `plot_h2_2_vs_tm_simulation.py` | look for data in `src/partials/` and would save to `src/figs/` |
| Broken | `FIGURE_DIR` in `run_disorder_avg.py` | resolves to `src/figs/`, so it needs fixing alongside the import |
| Broken | `src/zaratan/lightcone_cluster_plan.py` | writes its task table under `src/slurm/` |
| Broken | `slurm/` jobs | call `src/lightcone_cluster_*.py`, now in `src/zaratan/` |
| Broken | `tests/` | an installed package provides its own top-level `tests` module, which hides this repo's, so collection fails with `No module named 'tests.helpers'` |

Many scripts also add their own folder to `sys.path` before `import runs.…`. That
line no longer does anything useful, and the import works only because of the
editable install.

---

## Glossary

| Term | Meaning |
|---|---|
| **OTOC** | Out-of-time-order correlator; our measure of how far a disturbance has spread |
| **Echo (Loschmidt echo)** | Evolve forward, disturb, evolve backward |
| **Probe / perturbation site** | The qubit read out / the qubit disturbed |
| **`W` or `N`** | Number of qubits in one circuit |
| **`L` or `t`** | Number of layers in `U`; the circuit's depth, used as time |
| **`p`** | Probability of measuring each eligible qubit in each layer |
| **Haar-random** | Drawn uniformly from all unitaries |
| **Disorder realization** | One draw of all the circuit's random choices |
| **Shot** | One execution of a circuit, giving one outcome |
| **Light cone** | The region a disturbance can possibly have reached |
| **Front** | Where `C` actually rises, inside the light cone |
| **Scrambling** | Local information spreading until it can't be read locally |
| **Self-averaging** | A quantity's relative fluctuations shrinking as the system grows |
| **Transfer matrix (TM)** | The classical rule giving the exact disorder-averaged `C` |
| **Tiling** | Packing many realizations into one hardware circuit |
| **HQC** | Quantinuum's unit of hardware cost |
| **Nexus** | Quantinuum's cloud service for submitting jobs |
| **Null test** | A check where the answer is known in advance, used to catch errors |
