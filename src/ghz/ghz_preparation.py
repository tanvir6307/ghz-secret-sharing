"""
Realistic GHZ state preparation with gate-by-gate error tracking.

This module simulates GHZ preparation using density matrix evolution,
applying noise after each gate to track fidelity decay through the
CNOT cascade. This provides:
1. Fidelity trajectory (fidelity after each gate)
2. Per-gate error contributions
3. Error budget decomposition
4. Comparison with theoretical models
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, state_fidelity, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .ghz_circuit import (
    build_ghz_circuit,
    get_ideal_ghz_statevector,
    get_ideal_ghz_density_matrix,
)


def realistic_ghz_preparation(n_qubits, noise_model_obj, params,
                                shots=8192, method="density_matrix"):
    """
    Prepare an n-qubit GHZ state with realistic noise and track fidelity.

    Performs gate-by-gate density matrix simulation to extract fidelity
    trajectory and error budget.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in GHZ state.
    noise_model_obj : MultiQubitGHZNoise
        Noise model object (from composite_noise).
    params : dict
        Device parameters.
    shots : int
        Number of measurement shots for sampling.
    method : str
        Simulation method: "density_matrix" or "statevector".

    Returns
    -------
    dict
        Results containing:
        - 'circuit': The GHZ circuit
        - 'fidelity_trajectory': list of fidelities after each gate
        - 'error_budget': per-gate error contributions
        - 'final_fidelity': final GHZ state fidelity
        - 'gate_sequence': list of gate descriptions
        - 'measurement_counts': raw measurement results
        - 'density_matrix': final density matrix (if density_matrix method)
    """
    qubits_used = list(range(n_qubits))
    noise_model = noise_model_obj.build_noise_model(qubits_used=qubits_used)

    # Build gate sequence for tracking
    gate_sequence = _build_gate_sequence(n_qubits)

    # Gate-by-gate fidelity tracking via density matrix simulation
    fidelity_trajectory = []
    error_contributions = []
    ideal_ghz_sv = get_ideal_ghz_statevector(n_qubits)

    # Simulate step-by-step
    for step_idx in range(len(gate_sequence)):
        partial_qc = _build_partial_circuit(n_qubits, step_idx + 1,
                                             gate_sequence)
        partial_qc.save_density_matrix()

        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        result = sim.run(partial_qc, shots=1).result()
        rho = result.data()["density_matrix"]

        # Convert to numpy if needed
        if hasattr(rho, "data"):
            rho_np = np.array(rho.data)
        else:
            rho_np = np.array(rho)

        # Compute fidelity with ideal GHZ state
        fid = _compute_ghz_fidelity(rho_np, ideal_ghz_sv)
        fidelity_trajectory.append(fid)

        if step_idx == 0:
            error_contributions.append(1.0 - fid)
        else:
            error_contributions.append(
                max(fidelity_trajectory[-2] - fid, 0)
            )

    # Full circuit with measurements for counts
    meas_qc = build_ghz_circuit(n_qubits, barriers=False)
    meas_qc.measure_all()

    sim_meas = AerSimulator(method="automatic", noise_model=noise_model)
    meas_result = sim_meas.run(meas_qc, shots=shots).result()
    counts = meas_result.get_counts()

    # Calculate protocol timing
    t_h = params["gate_duration_h"]
    t_cnot = params["gate_duration_cnot"]
    time_points = [t_h * 1e9]  # H gate
    for i in range(n_qubits - 1):
        time_points.append(time_points[-1] + t_cnot * 1e9)

    return {
        "circuit": build_ghz_circuit(n_qubits),
        "fidelity_trajectory": fidelity_trajectory,
        "error_contributions": error_contributions,
        "final_fidelity": fidelity_trajectory[-1] if fidelity_trajectory else 0,
        "gate_sequence": gate_sequence,
        "time_points_ns": time_points,
        "measurement_counts": counts,
        "n_qubits": n_qubits,
    }


def _build_gate_sequence(n_qubits):
    """Build the ordered gate sequence for GHZ preparation."""
    sequence = []
    # Step 0: Hadamard on qubit 0
    sequence.append({
        "gate_type": "H",
        "qubits": [0],
        "step": 0,
        "description": "H(0): Create superposition",
    })
    # Steps 1 to n-1: CNOT cascade
    for i in range(n_qubits - 1):
        sequence.append({
            "gate_type": "CNOT",
            "qubits": [i, i + 1],
            "step": i + 1,
            "description": f"CNOT({i},{i+1}): Entangle qubit {i+1}",
        })
    return sequence


def _build_partial_circuit(n_qubits, num_gates, gate_sequence):
    """Build a circuit with only the first num_gates gates."""
    qc = QuantumCircuit(n_qubits)

    for i in range(min(num_gates, len(gate_sequence))):
        gate = gate_sequence[i]
        if gate["gate_type"] == "H":
            qc.h(gate["qubits"][0])
        elif gate["gate_type"] == "CNOT":
            qc.cx(gate["qubits"][0], gate["qubits"][1])

    return qc


def _compute_ghz_fidelity(rho, ideal_sv):
    """
    Compute fidelity between density matrix and ideal GHZ state.

    F = <GHZ|rho|GHZ>
    """
    fid = np.real(ideal_sv.conj() @ rho @ ideal_sv)
    return float(np.clip(fid, 0, 1))


def calculate_expected_fidelity(n_qubits, params):
    """
    Calculate expected GHZ fidelity from analytical error model.

    F_n ~ (1 - e_H) * (1 - e_CNOT)^(n-1) * exp(-t/T1) * exp(-t/T2)
         * (1 - leak_CNOT)^(n-1) * F_crosstalk

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    params : dict
        Device parameters.

    Returns
    -------
    dict
        Expected fidelity and breakdown.
    """
    n_cnots = n_qubits - 1

    # Gate errors
    F_H = 1 - params["single_qubit_error"]
    F_CNOT = (1 - params["cnot_error_mean"]) ** n_cnots

    # Thermal relaxation
    t_total = (params["gate_duration_h"]
               + n_cnots * params["gate_duration_cnot"])
    F_T1 = np.exp(-t_total / params["T1"])
    F_T2 = np.exp(-t_total / params["T2"])

    # Leakage
    F_leakage = (1 - params["leakage_cnot"]) ** n_cnots

    # Crosstalk (approximate)
    F_crosstalk = 0.995 ** n_cnots

    # Total
    F_total = F_H * F_CNOT * F_T1 * F_T2 * F_leakage * F_crosstalk

    return {
        "F_total": F_total,
        "F_H": F_H,
        "F_CNOT": F_CNOT,
        "F_T1": F_T1,
        "F_T2": F_T2,
        "F_leakage": F_leakage,
        "F_crosstalk": F_crosstalk,
        "t_total_us": t_total * 1e6,
        "n_cnots": n_cnots,
    }


def validate_against_experiment(sim_fidelity, exp_fidelity, exp_error,
                                 study_name=""):
    """
    Compare simulation fidelity against experimental benchmark.

    Parameters
    ----------
    sim_fidelity : float
        Our simulated fidelity.
    exp_fidelity : float
        Experimental fidelity.
    exp_error : float
        Experimental uncertainty.
    study_name : str
        Name of the experimental study.

    Returns
    -------
    dict
        Comparison results.
    """
    gap = abs(sim_fidelity - exp_fidelity)
    z_score = gap / exp_error if exp_error > 0 else float("inf")
    within_error = gap <= 2 * exp_error
    within_5pct = gap <= 0.05

    return {
        "study": study_name,
        "sim_fidelity": sim_fidelity,
        "exp_fidelity": exp_fidelity,
        "exp_error": exp_error,
        "gap": gap,
        "z_score": z_score,
        "within_2sigma": within_error,
        "within_5pct": within_5pct,
    }


# Experimental GHZ fidelity benchmarks from literature
EXPERIMENTAL_BENCHMARKS = {
    "Monz_2011_3q": {
        "study": "Monz et al. (2011)",
        "platform": "Trapped ions",
        "n_qubits": 3,
        "fidelity": 0.990,
        "error": 0.005,
        "citation": "PRL 106, 130506",
    },
    "Monz_2011_6q": {
        "study": "Monz et al. (2011)",
        "platform": "Trapped ions",
        "n_qubits": 6,
        "fidelity": 0.885,
        "error": 0.004,
        "citation": "PRL 106, 130506",
    },
    "Monz_2011_14q": {
        "study": "Monz et al. (2011)",
        "platform": "Trapped ions",
        "n_qubits": 14,
        "fidelity": 0.510,
        "error": 0.005,
        "citation": "PRL 106, 130506",
    },
    "Song_2019_12q": {
        "study": "Song et al. (2019)",
        "platform": "Superconducting",
        "n_qubits": 12,
        "fidelity": 0.550,
        "error": 0.020,
        "citation": "Science 365, 574",
    },
    "Song_2019_18q": {
        "study": "Song et al. (2019)",
        "platform": "Superconducting",
        "n_qubits": 18,
        "fidelity": 0.520,
        "error": 0.020,
        "citation": "Science 365, 574",
    },
    "Omran_2019_4q": {
        "study": "Omran et al. (2019)",
        "platform": "Rydberg atoms",
        "n_qubits": 4,
        "fidelity": 0.940,
        "error": 0.010,
        "citation": "Science 365, 570",
    },
    "Omran_2019_8q": {
        "study": "Omran et al. (2019)",
        "platform": "Rydberg atoms",
        "n_qubits": 8,
        "fidelity": 0.850,
        "error": 0.015,
        "citation": "Science 365, 570",
    },
    "Omran_2019_20q": {
        "study": "Omran et al. (2019)",
        "platform": "Rydberg atoms",
        "n_qubits": 20,
        "fidelity": 0.540,
        "error": 0.025,
        "citation": "Science 365, 570",
    },
    "IBM_3q_manila": {
        "study": "IBM Quantum (2024)",
        "platform": "Superconducting",
        "n_qubits": 3,
        "fidelity": 0.875,
        "error": 0.015,
        "citation": "IBM Quantum Experience",
    },
    "IBM_5q_manila": {
        "study": "IBM Quantum (2024)",
        "platform": "Superconducting",
        "n_qubits": 5,
        "fidelity": 0.780,
        "error": 0.020,
        "citation": "IBM Quantum Experience",
    },
    "Mooney_2021_3q": {
        "study": "Mooney et al. (2021)",
        "platform": "Superconducting (IBM)",
        "n_qubits": 3,
        "fidelity": 0.870,
        "error": 0.010,
        "citation": "J. Phys. Commun. 5, 095004",
    },
    "Mooney_2021_5q": {
        "study": "Mooney et al. (2021)",
        "platform": "Superconducting (IBM)",
        "n_qubits": 5,
        "fidelity": 0.750,
        "error": 0.015,
        "citation": "J. Phys. Commun. 5, 095004",
    },
    "Mooney_2021_7q": {
        "study": "Mooney et al. (2021)",
        "platform": "Superconducting (IBM)",
        "n_qubits": 7,
        "fidelity": 0.620,
        "error": 0.020,
        "citation": "J. Phys. Commun. 5, 095004",
    },
}
