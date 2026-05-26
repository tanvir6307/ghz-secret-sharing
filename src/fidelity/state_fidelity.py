"""
State and process fidelity computations for GHZ states.

Provides fidelity calculations specialized for multi-qubit GHZ states,
including:
- State fidelity (simulation vs. ideal GHZ)
- Entanglement witness measurement
- GHZ fidelity from measurement statistics
"""

import numpy as np
from qiskit.quantum_info import state_fidelity, DensityMatrix


def ghz_state_fidelity(rho, n_qubits):
    """
    Compute fidelity of a density matrix with the ideal n-qubit GHZ state.

    F = <GHZ_n| rho |GHZ_n>

    Parameters
    ----------
    rho : np.ndarray or DensityMatrix
        Density matrix (2^n x 2^n).
    n_qubits : int
        Number of qubits.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    if isinstance(rho, DensityMatrix):
        rho = np.array(rho.data)

    dim = 2**n_qubits
    if rho.shape != (dim, dim):
        raise ValueError(
            f"Density matrix shape {rho.shape} incompatible with {n_qubits} qubits"
        )

    # Ideal GHZ state vector
    ghz = np.zeros(dim, dtype=complex)
    ghz[0] = 1 / np.sqrt(2)
    ghz[dim - 1] = 1 / np.sqrt(2)

    fid = np.real(ghz.conj() @ rho @ ghz)
    return float(np.clip(fid, 0, 1))


def ghz_fidelity_from_counts(counts, n_qubits, total_shots=None):
    """
    Estimate GHZ fidelity from measurement counts (Z-basis only).

    For an ideal GHZ state measured in Z basis, only |00...0> and |11...1>
    should appear with equal probability. The Z-basis fidelity bound:
    F_Z >= P(00...0) + P(11...1) - 1 + 1/(2^(n-1))

    A simpler lower bound:
    F_lower >= (P(00...0) + P(11...1))

    Parameters
    ----------
    counts : dict
        Measurement counts from circuit execution.
    n_qubits : int
        Number of qubits.
    total_shots : int, optional
        Total number of shots. If None, computed from counts.

    Returns
    -------
    dict
        Fidelity estimates and statistics.
    """
    if total_shots is None:
        total_shots = sum(counts.values())

    # Target outcomes for GHZ
    all_zeros = "0" * n_qubits
    all_ones = "1" * n_qubits

    p_zeros = counts.get(all_zeros, 0) / total_shots
    p_ones = counts.get(all_ones, 0) / total_shots

    # GHZ population (probability in GHZ subspace)
    ghz_population = p_zeros + p_ones

    # Parity of all other outcomes
    error_population = 1 - ghz_population

    # Fidelity lower bound (from Z-basis only)
    f_lower = ghz_population

    # Balance: should be 50/50 in ideal case
    if ghz_population > 0:
        balance = min(p_zeros, p_ones) / max(p_zeros, p_ones)
    else:
        balance = 0

    return {
        "f_lower_bound": f_lower,
        "p_all_zeros": p_zeros,
        "p_all_ones": p_ones,
        "ghz_population": ghz_population,
        "error_population": error_population,
        "balance": balance,
        "total_shots": total_shots,
    }


def entanglement_witness(counts, n_qubits, total_shots=None):
    """
    Evaluate GHZ entanglement witness.

    The witness for n-qubit GHZ is:
    W = I/2 - |GHZ><GHZ|

    Tr(W * rho) < 0 implies genuine n-partite entanglement.

    From Z-basis measurements, we can bound:
    <W> <= 1/2 - (P(00...0) + P(11...1))

    Parameters
    ----------
    counts : dict
        Z-basis measurement counts.
    n_qubits : int
        Number of qubits.
    total_shots : int, optional
        Total shots.

    Returns
    -------
    dict
        Witness evaluation results.
    """
    fid_data = ghz_fidelity_from_counts(counts, n_qubits, total_shots)
    witness_value = 0.5 - fid_data["ghz_population"]

    return {
        "witness_value": witness_value,
        "is_entangled": witness_value < 0,
        "ghz_population": fid_data["ghz_population"],
        "threshold": 0.5,
        "margin": -witness_value if witness_value < 0 else 0,
    }


def coherence_from_x_basis(counts_x_basis, n_qubits, total_shots=None):
    """
    Estimate GHZ off-diagonal coherence from X-basis measurements.

    For |GHZ> = (|00...0> + |11...1>)/sqrt(2), the off-diagonal element
    <00...0|rho|11...1> can be estimated from X-basis measurements.

    For even n: <X^(n)> = 2 * Re(<00..0|rho|11..1>)
    This gives the GHZ coherence.

    Parameters
    ----------
    counts_x_basis : dict
        Measurement counts in X-basis (all qubits measured in X).
    n_qubits : int
        Number of qubits.
    total_shots : int, optional
        Total shots.

    Returns
    -------
    dict
        Coherence estimate.
    """
    if total_shots is None:
        total_shots = sum(counts_x_basis.values())

    # In X basis, parity should be even for ideal GHZ
    even_parity_count = 0
    odd_parity_count = 0

    for outcome, count in counts_x_basis.items():
        # Count number of 1s
        parity = outcome.count("1") % 2
        if parity == 0:
            even_parity_count += count
        else:
            odd_parity_count += count

    p_even = even_parity_count / total_shots
    p_odd = odd_parity_count / total_shots

    # Coherence = |<X^n>| = |P(even) - P(odd)|
    coherence = abs(p_even - p_odd)

    return {
        "coherence": coherence,
        "p_even_parity": p_even,
        "p_odd_parity": p_odd,
        "parity_contrast": p_even - p_odd,
    }


def full_ghz_fidelity_estimate(counts_z, counts_x, n_qubits):
    """
    Estimate GHZ fidelity from Z-basis and X-basis measurements.

    F_GHZ = (P(00..0) + P(11..1))/2 + coherence/2

    Parameters
    ----------
    counts_z : dict
        Z-basis measurement counts.
    counts_x : dict
        X-basis measurement counts.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    dict
        Full fidelity estimate.
    """
    z_data = ghz_fidelity_from_counts(counts_z, n_qubits)
    x_data = coherence_from_x_basis(counts_x, n_qubits)

    # Full fidelity estimate
    fidelity = z_data["ghz_population"] / 2 + x_data["coherence"] / 2

    return {
        "fidelity": fidelity,
        "z_component": z_data["ghz_population"] / 2,
        "x_component": x_data["coherence"] / 2,
        "z_basis_data": z_data,
        "x_basis_data": x_data,
    }
