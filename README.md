# GHZ-Based Quantum Secret Sharing

**Comprehensive noise-aware simulation and hardware validation of GHZ-based quantum secret sharing on superconducting processors.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit 2.x](https://img.shields.io/badge/Qiskit-2.x-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository provides a density-matrix simulation framework for the Hillery-Buzek-Berthiaume (HBB) quantum secret sharing protocol using Greenberger-Horne-Zeilinger (GHZ) states. The simulation incorporates **eight physically motivated noise channels** calibrated against IBM Quantum Falcon-class processors and is validated against measurements on the 156-qubit IBM Marrakesh (Heron-class) processor.

### Key Results

| Metric | Value |
|--------|-------|
| 3-party GHZ fidelity | 0.950 +/- 0.002 |
| Protocol success rate | 89.7% +/- 0.2% |
| Fidelity scaling | F(n) = 0.357 e^(-0.330n) + 0.819 (R^2 = 0.997) |
| Hardware validation gap (RMS) | 1.9% across n = 3, 5, 7 |
| Parties supported | n = 3 to 7 |

## Noise Model

The composite noise model includes eight independent channels:

1. **Thermal relaxation** (T1/T2 decay)
2. **Depolarizing gate errors** (single-qubit and CNOT)
3. **Coherent over-rotation** (systematic gate angle errors)
4. **ZZ crosstalk** (residual qubit-qubit coupling)
5. **Leakage** (transitions to non-computational states)
6. **1/f dephasing** (low-frequency flux noise)
7. **Collective correlated dephasing** (spatially correlated noise)
8. **SPAM errors** (state preparation and measurement)

All parameters are anchored to independently calibrated data from IBM Quantum hardware (see `parameters/justified_parameters.json`).

## Project Structure

```
ghz_secret_sharing/
├── src/                        # Core simulation library
│   ├── ghz/                    # GHZ state preparation circuits
│   │   ├── ghz_circuit.py      # Quantum circuit construction
│   │   └── ghz_preparation.py  # Density-matrix state preparation
│   ├── noise_models/           # Eight noise channel implementations
│   │   ├── markovian_noise.py      # T1/T2 thermal relaxation
│   │   ├── coherent_errors.py      # Gate over-rotations
│   │   ├── crosstalk_noise.py      # ZZ coupling crosstalk
│   │   ├── leakage_noise.py        # Leakage to |2> states
│   │   ├── one_over_f_noise.py     # 1/f^alpha dephasing
│   │   ├── collective_dephasing.py # Correlated dephasing
│   │   ├── spam_errors.py          # SPAM error modeling
│   │   └── composite_noise.py      # Combined 8-channel model
│   ├── secret_sharing/         # HBB protocol implementation
│   │   ├── hbb_protocol.py     # Protocol executor
│   │   └── security_analysis.py # Eavesdropping attack simulation
│   ├── fidelity/               # Fidelity estimation
│   │   └── state_fidelity.py   # Uhlmann-Jozsa fidelity
│   ├── validation/             # Hardware comparison tools
│   │   └── hardware_validation.py
│   └── utils/                  # Utilities
│       ├── data_export.py      # CSV/JSON export
│       ├── device_parameters.py # Device config loader
│       ├── error_budget.py     # Error decomposition
│       ├── statistical_tests.py # Bootstrap CI, KS tests
│       └── visualization.py    # Publication figure generation
├── tests/                      # Test suite
│   └── test_ghz_secret_sharing.py
├── parameters/                 # Calibrated device parameters
│   └── justified_parameters.json
├── data/simulation_results/    # All simulation output (CSV)
│   ├── ghz_preparation/
│   ├── error_budget/
│   ├── secret_sharing/
│   ├── security_analysis/
│   ├── scaling_analysis/
│   ├── statistical_analysis/
│   ├── hardware_validation/
│   └── literature_comparison/
├── figures/                    # Publication-quality figures (PDF)
├── output/                     # Hardware validation results
├── run_simulation.py           # Full simulation pipeline
├── run_hardware_validation.py  # IBM Quantum hardware execution
├── manuscript.tex              # Research manuscript (revtex4-2)
└── references.bib              # Bibliography (43 entries)
```

## Installation

### Prerequisites

- Python 3.10 or later
- An IBM Quantum account (only for hardware validation)

### Setup

```bash
git clone https://github.com/<your-username>/GHZ_secret_sharing.git
cd GHZ_secret_sharing
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---------|---------|
| Qiskit | >= 2.0 |
| Qiskit Aer | >= 0.15 |
| NumPy | >= 1.24 |
| SciPy | >= 1.10 |
| Matplotlib | >= 3.7 |

If no `requirements.txt` exists, install manually:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib
```

## Usage

### Run the Full Simulation

```bash
cd ghz_secret_sharing
python run_simulation.py
```

This executes the complete research pipeline (~525 simulated minutes of computation):

1. GHZ state preparation with fidelity tracking (n = 3 to 7)
2. Per-channel error budget analysis
3. HBB secret sharing protocol execution
4. Security analysis (intercept-resend and entangle-measure attacks)
5. Exponential scaling fit
6. Noise model comparison (individual vs. composite)
7. Monte Carlo convergence analysis
8. CSV data export and publication figure generation

All results are saved to `data/simulation_results/` as CSV files.

### Run Hardware Validation

Requires an IBM Quantum account with access to IBM Marrakesh (or another backend):

```bash
python run_hardware_validation.py
```

Results are saved to `output/`.

### Run Tests

```bash
python -m pytest tests/ -v
```

## Figures

The simulation generates the following publication-quality figures in `figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_ghz_scaling.pdf` | GHZ fidelity vs. number of parties with exponential fit |
| `fig2_error_budget.pdf` | Per-channel error contribution breakdown |
| `fig3_error_budget_pie.pdf` | Error budget pie chart |
| `fig4_secret_sharing.pdf` | HBB protocol success rate vs. party count |
| `fig5_security_threshold.pdf` | Security threshold: fidelity vs. noise scale |
| `noise_model_comparison.pdf` | Individual vs. composite noise model fidelities |
| `monte_carlo_convergence.pdf` | Fidelity convergence with trial count |
| `fig_hardware_validation.pdf` | Simulation vs. IBM Marrakesh comparison |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Hassan2026,
  author  = {Hassan, Tanvir},
  title   = {Comprehensive Noise-Aware Simulation and Hardware Validation
             of {GHZ}-Based Quantum Secret Sharing on Superconducting
             Processors},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

**Tanvir Hassan**
Department of Physics, Jagannath University, Dhaka 1100, Bangladesh
Email: tanvir6307@gmail.com
