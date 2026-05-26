"""
State Preparation and Measurement (SPAM) errors.

SPAM errors include:
1. State preparation: thermal population of |1> at initialization.
2. Measurement: readout confusion (misidentifying |0> as |1> or vice versa).

For GHZ secret sharing, all n parties must be measured, making readout
errors scale linearly with party count.

References:
    - IBM Quantum calibration data
    - Gambetta et al. (2007) PRA
"""

import numpy as np
from qiskit_aer.noise import ReadoutError
from qiskit_aer.noise.errors import pauli_error


def build_state_prep_error(thermal_population):
    """
    Build a state preparation error (thermal excitation).

    Parameters
    ----------
    thermal_population : float
        Probability of being in |1> after reset.

    Returns
    -------
    QuantumError or None
    """
    if thermal_population <= 0:
        return None
    return pauli_error([("X", thermal_population), ("I", 1 - thermal_population)])


def build_readout_error(error_rate):
    """
    Build a symmetric readout error.

    Parameters
    ----------
    error_rate : float
        Probability of misclassification.

    Returns
    -------
    ReadoutError
    """
    return ReadoutError(
        [[1 - error_rate, error_rate],
         [error_rate, 1 - error_rate]]
    )


def build_asymmetric_readout_error(p_0_given_1, p_1_given_0):
    """
    Build an asymmetric readout error.

    Parameters
    ----------
    p_0_given_1 : float
        P(measure 0 | state is 1).
    p_1_given_0 : float
        P(measure 1 | state is 0).

    Returns
    -------
    ReadoutError
    """
    return ReadoutError(
        [[1 - p_1_given_0, p_1_given_0],
         [p_0_given_1, 1 - p_0_given_1]]
    )


def ghz_spam_infidelity(n_qubits, thermal_population, readout_errors):
    """
    Estimate SPAM infidelity for n-qubit GHZ measurement.

    In GHZ secret sharing, all n qubits are initialized and measured.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    thermal_population : float
        State prep error probability.
    readout_errors : list[float]
        Readout error for each qubit.

    Returns
    -------
    dict
        SPAM error budget breakdown.
    """
    prep_error = n_qubits * thermal_population
    meas_error = sum(readout_errors[:n_qubits])
    total = prep_error + meas_error

    return {
        "state_prep_total": prep_error,
        "state_prep_per_qubit": thermal_population,
        "readout_total": meas_error,
        "readout_per_qubit": readout_errors[:n_qubits],
        "readout_mean": meas_error / n_qubits if n_qubits > 0 else 0,
        "total_spam": total,
        "n_qubits": n_qubits,
    }
