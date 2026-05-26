"""
Leakage noise model for transmon qubits.

Transmon qubits are weakly anharmonic oscillators. During gate operations,
population can leak from {|0>, |1>} to higher levels (primarily |2>).
For GHZ states, leakage accumulates with each CNOT in the cascade.

References:
    - Rol et al. (2020) PRL
    - Wood & Gambetta (2018) PRA
"""

import numpy as np
from qiskit_aer.noise.errors import depolarizing_error


def build_leakage_error_single(leakage_rate):
    """
    Create a single-qubit leakage error modeled as depolarizing noise.

    Parameters
    ----------
    leakage_rate : float
        Probability of leakage per gate (~0.0005 for single-qubit).

    Returns
    -------
    QuantumError or None
    """
    if leakage_rate <= 0:
        return None
    return depolarizing_error(leakage_rate, 1)


def build_leakage_error_two_qubit(leakage_rate):
    """
    Create a two-qubit leakage error for CNOT gates.

    Parameters
    ----------
    leakage_rate : float
        Probability of leakage per CNOT gate (~0.005).

    Returns
    -------
    QuantumError or None
    """
    if leakage_rate <= 0:
        return None
    return depolarizing_error(leakage_rate, 2)


def leakage_infidelity_contribution(num_cnots, leakage_rate_cnot,
                                     num_single_gates=0,
                                     leakage_rate_single=0):
    """
    Estimate total leakage contribution to infidelity.

    Parameters
    ----------
    num_cnots : int
        Number of CNOT gates.
    leakage_rate_cnot : float
        Leakage rate per CNOT.
    num_single_gates : int
        Number of single-qubit gates.
    leakage_rate_single : float
        Leakage rate per single-qubit gate.

    Returns
    -------
    float
        Estimated infidelity from leakage.
    """
    return (num_cnots * leakage_rate_cnot
            + num_single_gates * leakage_rate_single)


def ghz_leakage_infidelity(n_qubits, leakage_cnot, leakage_single):
    """
    Estimate leakage infidelity for n-qubit GHZ preparation.

    GHZ preparation uses 1 H gate + (n-1) CNOT gates.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    leakage_cnot : float
        Leakage per CNOT.
    leakage_single : float
        Leakage per single-qubit gate.

    Returns
    -------
    dict
        Leakage breakdown.
    """
    n_cnots = n_qubits - 1
    n_single = 1  # Hadamard gate

    single_contrib = n_single * leakage_single
    cnot_contrib = n_cnots * leakage_cnot
    total = single_contrib + cnot_contrib

    return {
        "single_qubit_leakage": single_contrib,
        "cnot_leakage": cnot_contrib,
        "total": total,
        "n_cnots": n_cnots,
        "n_single_gates": n_single,
    }
