"""
tests/helpers.py

Single location for loading src/__main__.py and re-exporting every
symbol the test suite needs.

Because src/__main__.py uses the reserved module name '__main__', a
plain `import __main__` inside a test would import the test runner
itself. importlib lets us load it by file path under a safe alias.

Every test file does:
    from tests.helpers import <whatever>
"""

import importlib.util
import pathlib

_src  = pathlib.Path(__file__).parent.parent / "src" / "__main__.py"
_spec = importlib.util.spec_from_file_location("schmetterling", _src)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# ── Re-export every symbol under test ─────────────────────────────────────────
calculate_hamming_distance_pdf          = _mod.calculate_hamming_distance_pdf
validate_nexus_connection               = _mod.validate_nexus_connection
generate_time_reversal_breaking_random_brick_wall = (
    _mod.generate_time_reversal_breaking_random_brick_wall
)
_compile_and_run                        = _mod._compile_and_run
run_time_sweep                          = _mod.run_time_sweep
sweep_p_and_t                           = _mod.sweep_p_and_t
compute_F                               = _mod.compute_F
debug_after_measurement                 = _mod.debug_after_measurement
count_consecutive_tk2_after_measure     = _mod.count_consecutive_tk2_after_measure
_run_circuit_backend                    = _mod._run_circuit_backend
disorder_average_vs_N                   = _mod.disorder_average_vs_N
plot_results                            = _mod.plot_results
sweep_x_and_t                           = _mod.sweep_x_and_t
plot_otoc                               = _mod.plot_otoc

# Raw module reference (needed for patch targets)
mod = _mod
