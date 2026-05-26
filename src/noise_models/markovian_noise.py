"""
Markovian thermal relaxation noise (T1 amplitude damping, T2 phase damping).

Models the dominant decoherence channel in superconducting transmon qubits.
T1 describes energy relaxation (|1> -> |0> decay).
T2 describes total dephasing (pure dephasing + T1 contribution).

Adapted for multi-qubit GHZ state preparation where each qubit may have
different coherence times and the circuit depth scales with party count.

References:
    - Nielsen & Chuang (2010), Ch. 8
    - IBM Quantum calibration data
"""

import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import thermal_relaxation_error


def build_thermal_relaxation_noise(
    T1_per_qubit,
    T2_per_qubit,
    gate_duration_single_ns,
    gate_duration_cnot_ns,
    measurement_duration_ns,
    coupling_map,
    num_qubits,
):
    """
    Build a Qiskit NoiseModel with thermal relaxation on every gate.

    Parameters
    ----------
    T1_per_qubit : list[float]
        T1 times in seconds for each qubit.
    T2_per_qubit : list[float]
        T2 times in seconds for each qubit.
    gate_duration_single_ns : float
        Single-qubit gate duration in ns.
    gate_duration_cnot_ns : float
        CNOT gate duration in ns.
    measurement_duration_ns : float
        Measurement duration in ns.
    coupling_map : list[tuple]
        List of (control, target) pairs.
    num_qubits : int
        Total number of qubits.

    Returns
    -------
    NoiseModel
        Qiskit noise model with thermal relaxation errors.
    """
    noise_model = NoiseModel()

    t_single = gate_duration_single_ns * 1e-9
    t_cnot = gate_duration_cnot_ns * 1e-9
    t_meas = measurement_duration_ns * 1e-9

    for q in range(num_qubits):
        T1 = T1_per_qubit[q] if q < len(T1_per_qubit) else T1_per_qubit[-1]
        T2 = T2_per_qubit[q] if q < len(T2_per_qubit) else T2_per_qubit[-1]
        T2 = min(T2, 2 * T1)

        # Single-qubit gate relaxation
        error_single = thermal_relaxation_error(T1, T2, t_single)
        noise_model.add_quantum_error(
            error_single,
            ["h", "x", "z", "s", "sdg", "sx", "rz", "ry", "rx", "id"],
            [q],
        )

        # Measurement relaxation
        error_meas = thermal_relaxation_error(T1, T2, t_meas)
        noise_model.add_quantum_error(error_meas, "measure", [q])

    # Two-qubit gate relaxation
    for pair in coupling_map:
        q0, q1 = pair
        T1_0 = T1_per_qubit[q0] if q0 < len(T1_per_qubit) else T1_per_qubit[-1]
        T2_0 = T2_per_qubit[q0] if q0 < len(T2_per_qubit) else T2_per_qubit[-1]
        T1_1 = T1_per_qubit[q1] if q1 < len(T1_per_qubit) else T1_per_qubit[-1]
        T2_1 = T2_per_qubit[q1] if q1 < len(T2_per_qubit) else T2_per_qubit[-1]
        T2_0 = min(T2_0, 2 * T1_0)
        T2_1 = min(T2_1, 2 * T1_1)

        error_cx = thermal_relaxation_error(T1_0, T2_0, t_cnot).expand(
            thermal_relaxation_error(T1_1, T2_1, t_cnot)
        )
        noise_model.add_quantum_error(error_cx, "cx", [q0, q1])

    return noise_model


def t1_decay_probability(duration, T1):
    """Compute amplitude damping probability gamma = 1 - exp(-t/T1)."""
    return 1 - np.exp(-duration / T1)


def t2_dephasing_rate(duration, T1, T2):
    """Compute pure dephasing rate gamma_phi."""
    rate_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
    if rate_phi < 0:
        rate_phi = 0
    return 1 - np.exp(-duration * rate_phi)


def ghz_thermal_infidelity(n_qubits, T1_per_qubit, T2_per_qubit,
                            gate_duration_h_s, gate_duration_cnot_s):
    """
    Estimate thermal relaxation contribution to GHZ state infidelity.

    For n-qubit GHZ: 1 H gate + (n-1) CNOT gates.
    Each qubit idles while other gates execute.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in GHZ state.
    T1_per_qubit : list[float]
        T1 for each qubit in seconds.
    T2_per_qubit : list[float]
        T2 for each qubit in seconds.
    gate_duration_h_s : float
        Hadamard gate duration in seconds.
    gate_duration_cnot_s : float
        CNOT gate duration in seconds.

    Returns
    -------
    dict
        Per-qubit and total thermal infidelity contributions.
    """
    total_duration = gate_duration_h_s + (n_qubits - 1) * gate_duration_cnot_s
    contributions = {}
    total_infid = 0.0

    for q in range(n_qubits):
        T1 = T1_per_qubit[q] if q < len(T1_per_qubit) else T1_per_qubit[-1]
        T2 = T2_per_qubit[q] if q < len(T2_per_qubit) else T2_per_qubit[-1]
        T2 = min(T2, 2 * T1)

        # Qubit q is idle for some portion of the circuit
        # Active during: H (if q==0), CNOT(q-1,q) and CNOT(q,q+1)
        # Idle for the rest
        idle_time = total_duration - gate_duration_cnot_s  # approximate
        if q == 0:
            idle_time = total_duration - gate_duration_h_s - gate_duration_cnot_s

        t1_infid = t1_decay_probability(total_duration, T1)
        t2_infid = t2_dephasing_rate(total_duration, T1, T2)

        contributions[f"qubit_{q}"] = {
            "T1_infidelity": t1_infid,
            "T2_infidelity": t2_infid,
            "idle_time_us": idle_time * 1e6,
        }
        total_infid += t1_infid + t2_infid

    contributions["total"] = total_infid / n_qubits  # Average per qubit
    return contributions
