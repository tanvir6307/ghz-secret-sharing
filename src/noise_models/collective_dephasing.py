"""
Collective dephasing noise model.

In multi-qubit systems, qubits can experience correlated dephasing
from shared noise sources (global magnetic field fluctuations, shared
control lines). For GHZ states, collective dephasing is particularly
important because GHZ states are maximally sensitive to correlated
phase noise - the superposition (|00...0> + |11...1>)/sqrt(2) acquires
a relative phase n times faster than a single qubit.

References:
    - Clemens et al. (2004) PRA
    - Chirolli & Burkard (2008) Advances in Physics
    - Monz et al. (2011) PRL
"""

import numpy as np
from qiskit_aer.noise.errors import phase_damping_error, depolarizing_error


def collective_dephasing_rate(T2_values, correlation=0.3):
    """
    Estimate collective dephasing rate for multiple qubits.

    Parameters
    ----------
    T2_values : list[float]
        T2 times for each qubit.
    correlation : float
        Noise correlation coefficient (0=independent, 1=fully correlated).

    Returns
    -------
    float
        Collective dephasing rate.
    """
    if len(T2_values) < 2:
        return 0.0
    gamma_avg = np.mean([1.0 / T2 for T2 in T2_values])
    return correlation * gamma_avg


def build_collective_dephasing_error(T2_q1, T2_q2, duration,
                                      correlation=0.3):
    """
    Build a two-qubit collective dephasing error.

    The correlated part of the dephasing is modelled as a two-qubit
    phase-damping channel whose strength is set by the *collective*
    dephasing rate ``gamma_c = correlation * mean(1/T2)``.

    Parameters
    ----------
    T2_q1, T2_q2 : float
        T2 times for the two qubits (seconds).
    duration : float
        Time window in seconds.
    correlation : float
        Correlation coefficient (already folded into gamma_c).

    Returns
    -------
    QuantumError or None
    """
    gamma_c = collective_dephasing_rate([T2_q1, T2_q2], correlation)
    p = 1 - np.exp(-gamma_c * duration)
    if p < 1e-8:
        return None
    # Use two single-qubit phase-damping errors (tensor product)
    # to avoid the full depolarizing channel which over-estimates
    # bit-flip noise.
    e1 = phase_damping_error(min(p, 1.0))
    e2 = phase_damping_error(min(p, 1.0))
    return e1.expand(e2)


def collective_dephasing_infidelity(T2_values, duration, correlation=0.3):
    """
    Estimate collective dephasing contribution to protocol infidelity.

    Parameters
    ----------
    T2_values : list[float]
        T2 for each qubit.
    duration : float
        Protocol duration in seconds.
    correlation : float
        Noise correlation.

    Returns
    -------
    float
        Estimated infidelity contribution.
    """
    if len(T2_values) < 2:
        return 0.0
    gamma_c = collective_dephasing_rate(T2_values, correlation)
    return 1 - np.exp(-gamma_c * duration)


def ghz_collective_dephasing_infidelity(n_qubits, T2_per_qubit,
                                         gate_duration_h_s,
                                         gate_duration_cnot_s,
                                         correlation=0.3):
    """
    Estimate collective dephasing infidelity for n-qubit GHZ.

    GHZ states are particularly sensitive because the N-qubit GHZ state
    |GHZ_N> = (|0>^N + |1>^N)/sqrt(2) dephases N times faster than a
    single qubit under collective noise.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    T2_per_qubit : list[float]
        T2 for each qubit.
    gate_duration_h_s : float
        H gate duration.
    gate_duration_cnot_s : float
        CNOT gate duration.
    correlation : float
        Noise correlation coefficient.

    Returns
    -------
    dict
        Collective dephasing breakdown.
    """
    total_duration = gate_duration_h_s + (n_qubits - 1) * gate_duration_cnot_s
    T2_used = T2_per_qubit[:n_qubits]

    # Standard collective dephasing
    standard_infid = collective_dephasing_infidelity(
        T2_used, total_duration, correlation
    )

    # GHZ enhancement factor: GHZ state dephases n times faster
    # The phase between |0...0> and |1...1> is n * phi
    # So the effective dephasing rate is n * gamma_collective
    gamma_c = collective_dephasing_rate(T2_used, correlation)
    ghz_enhanced_infid = 1 - np.exp(-n_qubits * gamma_c * total_duration)

    return {
        "standard_collective": standard_infid,
        "ghz_enhanced": ghz_enhanced_infid,
        "enhancement_factor": n_qubits,
        "correlation": correlation,
        "protocol_duration_us": total_duration * 1e6,
    }
