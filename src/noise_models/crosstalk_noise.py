"""
Cross-talk noise model: ZZ coupling between neighboring qubits.

In transmon qubits, residual ZZ coupling causes unwanted conditional phase
accumulation between idle qubits during gate operations. For GHZ states,
crosstalk is particularly important because the cascading CNOT structure
means spectator qubits accumulate phase errors during each gate.

References:
    - Malekakhlagh et al. (2020) PRX
    - Mundada et al. (2019) PRApplied
"""

import numpy as np
from qiskit.quantum_info import Operator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import coherent_unitary_error


def zz_unitary(theta):
    """
    Create the ZZ interaction unitary: exp(-i * theta/2 * ZZ).

    Parameters
    ----------
    theta : float
        Rotation angle = 2*pi*zz_rate*gate_time.

    Returns
    -------
    np.ndarray
        4x4 unitary matrix.
    """
    U = np.diag([
        np.exp(-1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(-1j * theta / 2),
    ])
    return U


def build_crosstalk_noise(zz_coupling_Hz, gate_duration_cnot_ns,
                           coupling_map, num_qubits):
    """
    Build crosstalk information for CNOT gates.

    During a CNOT on qubits (c, t), spectator qubits coupled to c or t
    experience a ZZ phase accumulation.

    Parameters
    ----------
    zz_coupling_Hz : dict
        ZZ coupling rates in Hz, keyed by "(q1,q2)" strings.
    gate_duration_cnot_ns : float
        CNOT gate duration in nanoseconds.
    coupling_map : list[tuple]
        Device coupling map.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    dict
        Maps CNOT qubit pairs to lists of (spectator, theta) pairs.
    """
    t_cnot_s = gate_duration_cnot_ns * 1e-9
    crosstalk_info = {}

    for pair in coupling_map:
        c, t = pair
        spectators = []
        for zz_key, zz_rate in zz_coupling_Hz.items():
            q1, q2 = _parse_qubit_pair(zz_key)
            if q1 is None:
                continue
            if c in (q1, q2) or t in (q1, q2):
                spectator = None
                if q1 not in (c, t) and q1 < num_qubits:
                    spectator = q1
                elif q2 not in (c, t) and q2 < num_qubits:
                    spectator = q2
                if spectator is not None:
                    theta = 2 * np.pi * abs(zz_rate) * t_cnot_s
                    spectators.append((spectator, theta))

        crosstalk_info[(c, t)] = spectators

    return crosstalk_info


def _parse_qubit_pair(key_str):
    """Parse a qubit pair string like '(0,1)' into (0, 1)."""
    try:
        cleaned = key_str.strip("()")
        parts = cleaned.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        return None, None


def crosstalk_phase_error(theta):
    """
    Convert a ZZ phase angle into process infidelity.

    1 - F_process = sin^2(theta/2)
    """
    return min(np.sin(theta / 2) ** 2, 1.0)


def ghz_crosstalk_infidelity(n_qubits, zz_coupling_Hz,
                              gate_duration_cnot_ns, coupling_map):
    """
    Estimate total crosstalk infidelity for GHZ preparation.

    For n-qubit GHZ, we have (n-1) CNOT gates. During each CNOT,
    spectator qubits accumulate ZZ phase errors.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in GHZ state.
    zz_coupling_Hz : dict
        ZZ coupling rates.
    gate_duration_cnot_ns : float
        CNOT duration in ns.
    coupling_map : list[tuple]
        Device coupling map.

    Returns
    -------
    dict
        Per-CNOT and total crosstalk contributions.
    """
    ct_info = build_crosstalk_noise(
        zz_coupling_Hz, gate_duration_cnot_ns, coupling_map, n_qubits
    )

    contributions = {}
    total = 0.0

    # GHZ uses CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1)
    for i in range(n_qubits - 1):
        c, t = i, i + 1
        pair = (c, t)
        spectators = ct_info.get(pair, [])
        pair_error = 0.0
        for spec, theta in spectators:
            if spec < n_qubits:
                pair_error += crosstalk_phase_error(theta)
        pair_error = min(pair_error, 1.0)
        contributions[f"CNOT({c},{t})"] = pair_error
        total += pair_error

    contributions["total"] = min(total, 1.0)
    return contributions
