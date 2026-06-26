# Schmetterling Effect

Disorder-averaged OTOC (out-of-time-order correlator) experiment in a
Haar-random brick-wall circuit with mid-circuit measurements.  The name
references the butterfly effect — *Schmetterling* is German for butterfly.

---

## Physics background

The experiment measures quantum information scrambling via a Loschmidt-echo
protocol:

```
|ψ⟩  →  state-prep  →  U  →  [perturbation]  →  U†  →  measure probe
```

`U` is a depth-`L` brick-wall circuit of Haar-random SU(4) gates.  Mid-circuit
measurements are sprinkled through `U` and `U†` at rate `p` per qubit per
layer.  The signal is:

```
C(t) = ⟨Z_probe⟩_unperturbed  −  ⟨Z_probe⟩_perturbed
```

- **p = 0**: unitary dynamics, fast scrambling — C(t) grows toward 1.
- **large p**: measurements localise information (quantum Zeno) — C(t) is
  suppressed.
- **phase transition**: between these two regimes at a critical p*.

The disorder average over gate angles, initial Pauli words, and measurement
sites reduces shot noise and isolates the universal scrambling signal.

---

## Repository layout

```
src/
  __main__.py              — core library: circuit generation, sweeps, plotting
  run_disorder_avg.py      — experiment orchestrator (config + parallel schemes)
  transfer_matrix_otoc.py  — exact analytical OTOC via Pauli-weight transfer matrix
notebooks/
  estimate_gate_count.ipynb
figs/                      — output figures (auto-created)
```

---

## Source files

### `src/__main__.py`

The core library.  All public functions are importable.

| Function | Purpose |
|---|---|
| `generate_time_reversal_breaking_random_brick_wall(...)` | Build one perturbed or unperturbed echo circuit |
| `_run_circuits(circuits, backend, n_shots, tags)` | Compile and batch-submit circuits to a pytket backend |
| `sweep_single_shot_disorder(...)` | Disorder average: each sample is a fresh independent circuit, 1 shot |
| `sweep_over_all_disorder_axes(...)` | Disorder average: fixed circuit realizations × `N_SHOTS` shots each |
| `compute_C_t(raw)` | Compute C, variance, standard error, CV from raw realization arrays |
| `plot_results(all_stats, ...)` | Save two figures: C vs T and CV vs T |
| `build_aer_noise_model(...)` | Build a Qiskit Aer noise model (depolarizing + T1/T2 + readout) |

#### Circuit structure

Each circuit has `W` qubits and `n_bits = 2` classical bits:
- **bit 0** (scratch): receives all mid-circuit measurement outcomes (overwritten, discarded)
- **bit 1** (probe): final readout of `probe_site`

State preparation randomises all qubits except `probe_site` with a random
Pauli ∈ {I, X, Y, Z}; `probe_site` gets `Ry(probe_angle·π)`.

Gates are Haar-random SU(4), decomposed natively as `TK1 · TK2 · TK1`
(pytket's KAK form).  The dagger U† is built by reversing layer order and
negating all angles.

#### Disorder axes (independent RNG streams)

| Axis | Seed formula |
|---|---|
| Gate angles | `BASE_SEED × 10 000 + c_idx` |
| Initial Pauli word | `BASE_SEED × 10 000 000 + c_idx × 1 000 + i_idx` |
| U measurement sites | `(MEAS_SEED+1) × 10 000 000 + c_idx × 1 000 + i_idx` |
| U† measurement sites | `(MEAS_SEED+2) × 10 000 000 + c_idx × 1 000 + i_idx` |

---

### `src/run_disorder_avg.py`

Experiment orchestrator.  Edit the config block at the top; all scheme
functions read from module-level globals.

#### Config parameters

**Circuit / sweep**

| Parameter | Default | Meaning |
|---|---|---|
| `P_VALUES` | `[0.0, 0.02]` | Measurement rates to sweep |
| `N_VALUES` | `[6]` | System sizes (qubits, must be even) |
| `L_MAX` | `20` | Max circuit depth |
| `L_VALUES` | `[2,5,10,20,30]` | Explicit depth list (overrides `L_MAX` + `RECORD_EVERY` when set; `None` to use range) |
| `RECORD_EVERY` | `4` | Step size when `L_VALUES` is `None` |
| `N_REALIZATIONS` | `1000` | Circuits per (N, p, L) in single-shot mode |
| `N_CIRCUITS` | `5` | Gate-angle seeds in multi-shot mode |
| `N_INIT_STATES` | `5` | Initial-state seeds in multi-shot mode |
| `N_SHOTS` | `100` | Shots per circuit in multi-shot mode |
| `BASE_SEED` | `42` | Root RNG seed |
| `MEAS_SEED` | `7` | Root measurement RNG seed |
| `PERT_OP` | `"X"` | Perturbation gate (`"X"/"Y"/"Z"/"H"/"S"/"T"` or `"measure"`) |
| `PROBE_ANGLE` | `0.0` | Ry half-turns on probe qubit |
| `DEVICE_NAME` | `None` | `None` → local AerBackend; `"H2-1LE"` → Quantinuum hardware |

**Noise model** (local AerBackend only; all `None` = ideal)

| Parameter | Default | Meaning |
|---|---|---|
| `NOISE_P1Q` | `None` | Single-qubit depolarizing error rate |
| `NOISE_P2Q` | `None` | Two-qubit depolarizing error rate |
| `NOISE_PM` | `None` | Symmetric readout bitflip probability |
| `NOISE_T1` | `None` | Longitudinal relaxation time T1 (seconds) |
| `NOISE_T2` | `None` | Transverse relaxation time T2 (seconds; must be ≤ 2·T1) |
| `NOISE_T_1Q` | `50e-9` | Single-qubit gate duration (seconds) |
| `NOISE_T_2Q` | `300e-9` | Two-qubit gate duration (seconds) |

Depolarizing and thermal-relaxation channels compose automatically when both
are set.

**Transfer-matrix overlay** (set `TM_NUM_QUBITS = None` to skip entirely)

| Parameter | Default | Meaning |
|---|---|---|
| `TM_NUM_QUBITS` | `6` | System size for TM run (`None` disables) |
| `TM_TF` | `30` | Number of time steps to evolve |
| `TM_PERT_SITE` | `3` | Perturbation qubit (0-indexed) |
| `TM_PROBE_SITE` | `0` | Probe qubit (0-indexed) |
| `TM_P_VALUES` | `None` | p values for TM (`None` mirrors `P_VALUES`) |
| `TM_BC_TYPE` | `"open"` | `"open"` or `"periodic"` boundary conditions |

#### Running

**Sequential (single process)**

```bash
# single-shot mode (default): N_REALIZATIONS independent circuits, 1 shot each
python src/run_disorder_avg.py

# multi-shot mode: N_CIRCUITS × N_INIT_STATES realizations, N_SHOTS shots each
python src/run_disorder_avg.py --mode multi_shot

# override L values from CLI
python src/run_disorder_avg.py --L-values 1,2,5,10,25
```

**Parallel (nohup)**

Four partitioning schemes are available. Print ready-to-paste launch commands:

```bash
python src/run_disorder_avg.py --help-nohup
```

| Scheme | Partition axis | Jobs | Command |
|---|---|---|---|
| 1 | System size N | `len(N_VALUES)` | `--scheme 1 --job-id {n_idx}` |
| 2 | Gate seed c_idx | `N_CIRCUITS` | `--scheme 2 --job-id {c_idx}` |
| 3 | Realization (c, i) | `N_CIRCUITS × N_INIT_STATES` | `--scheme 3 --job-id {c} --secondary-id {i}` |
| 4 | (N, p) grid point | `len(N_VALUES) × len(P_VALUES)` | `--scheme 4 --job-id {n} --secondary-id {p}` |

After all parallel jobs complete, merge and plot:

```bash
python src/run_disorder_avg.py --scheme 2 --merge
```

**Cost estimate (Quantinuum hardware)**

```bash
python src/run_disorder_avg.py --estimate-hqc
python src/run_disorder_avg.py --estimate-hqc --mode multi_shot
```

---

### `src/transfer_matrix_otoc.py`

Exact disorder-averaged OTOC via the classical Pauli-weight transfer matrix.
No Monte Carlo sampling — a single call gives the exact C(t) curve.

```python
from transfer_matrix_otoc import ButterflyTransferMatrix

tm = ButterflyTransferMatrix(num_qubits=6, bc_type="open")
t, C = tm.compute_otoc(p=0.0, tf=30)                        # defaults: pert=N//2, probe=0
t, C = tm.compute_otoc(p=0.02, tf=30, pert_site=3, probe_site=0)
```

The 4×4 local transfer matrix `T` encodes how a qubit pair (i, j) redistributes
Pauli weight under one Haar-averaged gate + measurements at rate `p`:

```
  00 → 00
  01 → a·00 + b·11
  10 → a·00 + b·11
  11 → c·00 + d·11

  a = (6 + 2p − p²) / 30
  b = 4(3 − 4p + 2p²) / 15
  c = p(8 − 8p + 4p² − p³) / 60
  d = (3 + 8(p−1)² + 4(p−1)⁴) / 15
```

**Pauli-agnosticism**: the TM averages over all Pauli directions — it does not
distinguish X, Y, or Z perturbations.  This is exact for the Haar-averaged
ensemble; differences between specific Paulis wash out in the disorder average.

**Factor of 2**: the TM result is scaled by 1/2 before overlay plotting to
match the normalisation convention of the quantum circuit experiment.

---

## Output figures

| File | Contents |
|---|---|
| `figs/C_vs_t_<DATE>.png` | Three-panel: C vs T (varying N), C vs p (varying N), C vs T (varying p) |
| `figs/cv_vs_t_<DATE>.png` | Same layout for CV = √Var / \|C\| (self-averaging diagnostic) |
| `figs/qc_vs_tm_<DATE>.png` | One panel per p: QC errorbars for all N overlaid with exact TM curve |

---

## Dependencies

- `pytket` — circuit IR and compilation
- `pytket-extensions-qiskit` — AerBackend (local simulation)
- `pytket-extensions-quantinuum` — QuantinuumBackend (hardware)
- `qiskit-aer` — Aer simulator + noise models
- `qnexus` — Quantinuum Nexus authentication
- `numpy`, `matplotlib`
