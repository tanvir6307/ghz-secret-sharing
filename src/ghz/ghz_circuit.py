"""
GHZ state circuit construction.

Builds quantum circuits for n-qubit GHZ state preparation:
|GHZ_n> = (|0...0> + |1...1>) / sqrt(2)

Uses a Hadamard on qubit 0 followed by a cascade of CNOT gates:
H(0) -> CNOT(0,1) -> CNOT(1,2) -> ... -> CNOT(n-2,n-1)

This module constructs the ideal circuits; noise is added separately
by the noise model.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def build_ghz_circuit(n_qubits, barriers=True, label=None):
    """
    Build an n-qubit GHZ state preparation circuit.

    |GHZ_n> = (|00...0> + |11...1>) / sqrt(2)

    Parameters
    ----------
    n_qubits : int
        Number of qubits (parties). Must be >= 2.
    barriers : bool
        Whether to insert barriers between stages.
    label : str, optional
        Circuit label.

    Returns
    -------
    QuantumCircuit
        GHZ preparation circuit (no measurements).

    Raises
    ------
    ValueError
        If n_qubits < 2.
    """
    if n_qubits < 2:
        raise ValueError(f"GHZ state requires at least 2 qubits, got {n_qubits}")

    if label is None:
        label = f"GHZ-{n_qubits}"

    qr = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qr, name=label)

    # Hadamard on qubit 0: |0> -> (|0> + |1>)/sqrt(2)
    qc.h(qr[0])

    if barriers:
        qc.barrier()

    # CNOT cascade: entangle qubits sequentially
    for i in range(n_qubits - 1):
        qc.cx(qr[i], qr[i + 1])

    if barriers:
        qc.barrier()

    return qc


def build_ghz_circuit_with_measurement(n_qubits, barriers=True):
    """
    Build a GHZ circuit with measurement on all qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    barriers : bool
        Insert barriers.

    Returns
    -------
    QuantumCircuit
    """
    qc = build_ghz_circuit(n_qubits, barriers=barriers)
    cr = ClassicalRegister(n_qubits, "c")
    qc.add_register(cr)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def build_ghz_tomography_circuits(n_qubits):
    """
    Build circuits for GHZ state tomography.

    Creates measurement circuits in all 3^n Pauli bases (X, Y, Z per qubit).
    For n > 4, uses a reduced set of diagonal + off-diagonal bases.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    list[QuantumCircuit]
        Tomography circuits.
    list[str]
        Basis labels for each circuit.
    """
    bases = ["Z", "X", "Y"]

    if n_qubits <= 4:
        # Full tomography
        import itertools

        basis_combos = list(itertools.product(bases, repeat=n_qubits))
    else:
        # Reduced tomography for large systems
        # Focus on: all-Z, all-X, all-Y, and pairwise XY bases
        basis_combos = []
        # All same-basis
        for b in bases:
            basis_combos.append(tuple([b] * n_qubits))
        # Pairwise variations
        for i in range(n_qubits):
            for b in bases:
                combo = ["Z"] * n_qubits
                combo[i] = b
                basis_combos.append(tuple(combo))

    circuits = []
    labels = []

    for combo in basis_combos:
        ghz_qc = build_ghz_circuit(n_qubits, barriers=True)
        cr = ClassicalRegister(n_qubits, "c")
        ghz_qc.add_register(cr)

        # Add measurement basis rotations
        for q, basis in enumerate(combo):
            if basis == "X":
                ghz_qc.h(q)
            elif basis == "Y":
                ghz_qc.sdg(q)
                ghz_qc.h(q)
            # Z basis: no rotation needed

        ghz_qc.measure(range(n_qubits), range(n_qubits))

        label = "".join(combo)
        ghz_qc.name = f"GHZ-{n_qubits}_tomo_{label}"
        circuits.append(ghz_qc)
        labels.append(label)

    return circuits, labels


def get_ideal_ghz_statevector(n_qubits):
    """
    Return the ideal GHZ state vector.

    |GHZ_n> = (|00...0> + |11...1>) / sqrt(2)

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        State vector of length 2^n.
    """
    dim = 2**n_qubits
    sv = np.zeros(dim, dtype=complex)
    sv[0] = 1 / np.sqrt(2)         # |00...0>
    sv[dim - 1] = 1 / np.sqrt(2)   # |11...1>
    return sv


def get_ideal_ghz_density_matrix(n_qubits):
    """
    Return the ideal GHZ state density matrix.

    rho_GHZ = |GHZ><GHZ|

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        Density matrix of shape (2^n, 2^n).
    """
    sv = get_ideal_ghz_statevector(n_qubits)
    return np.outer(sv, sv.conj())


def ghz_circuit_depth(n_qubits):
    """
    Calculate the circuit depth for GHZ preparation.

    Depth = 1 (H) + (n-1) (CNOTs) = n gates.
    But since CNOTs are sequential, circuit depth = n.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    dict
        Circuit resource requirements.
    """
    return {
        "n_qubits": n_qubits,
        "n_hadamard": 1,
        "n_cnot": n_qubits - 1,
        "total_gates": n_qubits,
        "circuit_depth": n_qubits,
        "two_qubit_depth": n_qubits - 1,
    }
