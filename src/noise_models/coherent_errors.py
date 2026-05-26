"""
Coherent control errors: systematic over/under-rotation.

Real quantum gates have small systematic calibration errors that
accumulate constructively (unlike stochastic noise). For GHZ states,
the cascading CNOT structure means coherent errors can interfere.

References:
    - Barends et al. (2014) Nature
    - McKay et al. (2017) PRA
"""

import numpy as np
from qiskit_aer.noise.errors import coherent_unitary_error


def over_rotation_unitary_rz(theta, epsilon):
    """Rz gate with over-rotation: Rz(theta*(1+epsilon))."""
    angle = theta * (1 + epsilon)
    return np.array([
        [np.exp(-1j * angle / 2), 0],
        [0, np.exp(1j * angle / 2)]
    ])


def over_rotation_unitary_rx(theta, epsilon):
    """Rx gate with over-rotation."""
    angle = theta * (1 + epsilon)
    return np.array([
        [np.cos(angle / 2), -1j * np.sin(angle / 2)],
        [-1j * np.sin(angle / 2), np.cos(angle / 2)]
    ])


def coherent_error_probability(epsilon):
    """
    Convert coherent over-rotation parameter to effective error probability.

    For small epsilon: error ~ epsilon^2.
    """
    return min(epsilon**2, 1.0)


def build_coherent_error_single_qubit(epsilon):
    """
    Build a coherent over-rotation error for single-qubit gates.

    Models the *deviation* from the intended gate as a small parasitic
    Z-rotation of angle ``pi * epsilon``.  For epsilon = 0.01 this
    corresponds to a 1.8° rotation and infidelity ~2.5e-4 per gate.

    Parameters
    ----------
    epsilon : float
        Over-rotation fraction.

    Returns
    -------
    QuantumError or None
    """
    if abs(epsilon) < 1e-10:
        return None
    # Only the error part: Rz(pi * epsilon)  (NOT the full gate)
    angle = np.pi * epsilon
    U = np.array([
        [np.exp(-1j * angle / 2), 0],
        [0, np.exp(1j * angle / 2)]
    ])
    return coherent_unitary_error(U)


def build_coherent_error_cnot(epsilon):
    """
    Build a coherent error for CNOT gates.

    Models systematic calibration error as a small parasitic Z-rotation
    of angle ``pi * epsilon`` on the target qubit after the CNOT.

    Parameters
    ----------
    epsilon : float
        Over-rotation fraction.

    Returns
    -------
    QuantumError or None
    """
    if abs(epsilon) < 1e-10:
        return None
    angle = np.pi * epsilon
    U_target = np.array([
        [np.exp(-1j * angle / 2), 0],
        [0, np.exp(1j * angle / 2)]
    ])
    I2 = np.eye(2)
    U_full = np.kron(I2, U_target)
    return coherent_unitary_error(U_full)


def ghz_coherent_infidelity(n_qubits, epsilon):
    """
    Estimate coherent error infidelity for n-qubit GHZ preparation.

    Coherent errors can accumulate constructively in the worst case.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    epsilon : float
        Over-rotation parameter.

    Returns
    -------
    dict
        Coherent error breakdown.
    """
    n_cnots = n_qubits - 1
    n_single = 1  # Hadamard

    # Coherent errors: worst case constructive interference
    single_err = coherent_error_probability(epsilon)
    cnot_err = coherent_error_probability(epsilon)

    # For constructive interference: errors add linearly in amplitude
    # then square for probability
    worst_case = (np.sqrt(single_err) + n_cnots * np.sqrt(cnot_err)) ** 2
    # Average case: errors add in quadrature
    avg_case = single_err + n_cnots * cnot_err

    return {
        "single_qubit_coherent": single_err,
        "cnot_coherent_per_gate": cnot_err,
        "total_average": min(avg_case, 1.0),
        "total_worst_case": min(worst_case, 1.0),
        "epsilon": epsilon,
    }
