"""
Error budget computation and formatting for GHZ states.

Provides tools to compute, format, and compare error budgets
across different party counts.
"""

import numpy as np


def compute_ghz_error_budget(n_qubits, params):
    """
    Compute analytical error budget for n-qubit GHZ preparation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    params : dict
        Device parameters.

    Returns
    -------
    dict
        Error budget with per-source contributions.
    """
    n_cnots = n_qubits - 1
    t_h = params["gate_duration_h"]
    t_cnot = params["gate_duration_cnot"]
    t_total = t_h + n_cnots * t_cnot

    budget = {}

    # 1. CNOT gate errors (dominant for multi-qubit)
    cnot_err = 1 - (1 - params["cnot_error_mean"]) ** n_cnots
    budget["CNOT_errors"] = cnot_err

    # 2. Single-qubit gate error (H gate)
    sq_err = params["single_qubit_error"]
    budget["H_gate_error"] = sq_err

    # 3. T1 relaxation
    t1_err = 1 - np.exp(-t_total / params["T1"])
    budget["T1_relaxation"] = t1_err * n_qubits

    # 4. T2 dephasing
    t2_err = 1 - np.exp(-t_total / params["T2"])
    budget["T2_dephasing"] = t2_err * n_qubits

    # 5. Readout errors (all qubits measured)
    read_err = n_qubits * params["readout_error_mean"]
    budget["readout"] = read_err

    # 6. State preparation
    prep_err = n_qubits * params["thermal_population"]
    budget["state_preparation"] = prep_err

    # 7. Leakage
    leak_err = n_cnots * params["leakage_cnot"] + params["leakage_single"]
    budget["leakage"] = leak_err

    # 8. Crosstalk (approximate)
    ct_err = n_cnots * 0.001  # ~0.1% per CNOT from ZZ coupling
    budget["crosstalk"] = ct_err

    # Total
    budget["total"] = sum(budget.values())

    return budget


def format_error_budget(budget, title="Error Budget"):
    """
    Format error budget as a printable table.

    Parameters
    ----------
    budget : dict
        Error budget.
    title : str
        Table title.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 55)
    lines.append(f"{'Source':<25} {'Contribution':>12} {'Percent':>10}")
    lines.append("-" * 55)

    for source, value in budget.items():
        if source == "total":
            lines.append("-" * 55)
        lines.append(
            f"{source:<25} {value:>12.6f} {value*100:>9.3f}%"
        )

    lines.append("=" * 55)
    return "\n".join(lines)


def compare_error_budgets(budgets_dict):
    """
    Compare error budgets across different party counts.

    Parameters
    ----------
    budgets_dict : dict[int, dict]
        Maps n_qubits -> error_budget.

    Returns
    -------
    dict
        Comparison data suitable for plotting.
    """
    all_sources = set()
    for budget in budgets_dict.values():
        all_sources.update(budget.keys())
    all_sources.discard("total")

    comparison = {
        "n_parties": sorted(budgets_dict.keys()),
        "sources": sorted(all_sources),
    }

    for source in sorted(all_sources):
        comparison[source] = [
            budgets_dict[n].get(source, 0)
            for n in comparison["n_parties"]
        ]

    comparison["total"] = [
        budgets_dict[n].get("total", 0)
        for n in comparison["n_parties"]
    ]

    return comparison


def cumulative_fidelity_decay(n_qubits, params):
    """
    Calculate cumulative fidelity after each CNOT in GHZ cascade.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    params : dict
        Device parameters.

    Returns
    -------
    list[dict]
        Fidelity after each gate step.
    """
    steps = []
    fidelity = 1.0

    # Step 0: Hadamard gate
    fidelity *= (1 - params["single_qubit_error"])
    steps.append({
        "step": 0,
        "gate": "H(0)",
        "fidelity": fidelity,
        "error": 1 - fidelity,
    })

    # Steps 1 to n-1: CNOT gates
    for i in range(n_qubits - 1):
        fidelity *= (1 - params["cnot_error_mean"])
        # Add thermal decay during this gate
        fidelity *= np.exp(-params["gate_duration_cnot"] / params["T1"])
        steps.append({
            "step": i + 1,
            "gate": f"CNOT({i},{i+1})",
            "fidelity": fidelity,
            "error": 1 - fidelity,
        })

    return steps
