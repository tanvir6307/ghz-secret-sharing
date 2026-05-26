"""
1/f noise model (non-Markovian dephasing).

Charge and flux noise in superconducting qubits exhibit a 1/f^alpha
power spectral density, causing non-Markovian dephasing beyond the
simple T2 model.

For GHZ states, 1/f noise is particularly relevant because the
preparation circuit can be long (multiple CNOT gates), allowing
significant low-frequency noise accumulation.

References:
    - Yan et al. (2016) Nature Communications
    - Ithier et al. (2005) PRB
    - Paladino et al. (2014) Rev. Mod. Phys.
"""

import numpy as np
from qiskit_aer.noise.errors import phase_damping_error


def generate_1f_noise_trajectory(duration_s, dt_s, alpha=0.9, amplitude=5e-6):
    """
    Generate a 1/f^alpha noise time series.

    Parameters
    ----------
    duration_s : float
        Total duration in seconds.
    dt_s : float
        Time step in seconds.
    alpha : float
        Spectral exponent (typically 0.7-1.1).
    amplitude : float
        Noise amplitude in energy units.

    Returns
    -------
    np.ndarray
        Noise trajectory.
    """
    n_steps = max(int(duration_s / dt_s), 2)
    white_noise = np.random.randn(n_steps)

    freqs = np.fft.fftfreq(n_steps, dt_s)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0

    psd_filter = 1.0 / np.abs(freqs) ** (alpha / 2.0)
    noise_fft = np.fft.fft(white_noise) * psd_filter
    noise = np.real(np.fft.ifft(noise_fft)) * amplitude

    return noise


def one_over_f_dephasing_rate(protocol_duration_s, alpha=0.9, amplitude=5e-6):
    """
    Estimate additional dephasing from 1/f noise beyond Markovian T2.

    Parameters
    ----------
    protocol_duration_s : float
        Total protocol duration in seconds.
    alpha : float
        Spectral exponent.
    amplitude : float
        Noise amplitude.

    Returns
    -------
    float
        Additional dephasing probability.
    """
    t_ir = 100e-6  # Infrared cutoff
    if protocol_duration_s <= 0:
        return 0.0

    ratio = protocol_duration_s / t_ir
    if ratio <= 0:
        return 0.0

    gamma_1f = amplitude**2 * abs(np.log(ratio + 1e-30)) * 1e6
    p_dephasing = min(1.0 - np.exp(-gamma_1f), 1.0)
    p_dephasing = max(p_dephasing, 0.0)
    return p_dephasing


def build_1f_dephasing_error(protocol_duration_s, alpha=0.9, amplitude=5e-6):
    """
    Build a phase damping error representing 1/f noise contribution.

    Returns
    -------
    QuantumError or None
    """
    p = one_over_f_dephasing_rate(protocol_duration_s, alpha, amplitude)
    if p < 1e-8:
        return None
    return phase_damping_error(p)


def ghz_1f_infidelity(n_qubits, gate_duration_h_s, gate_duration_cnot_s,
                       alpha=0.9, amplitude=5e-6):
    """
    Estimate 1/f noise infidelity for n-qubit GHZ preparation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    gate_duration_h_s : float
        H gate duration in seconds.
    gate_duration_cnot_s : float
        CNOT gate duration in seconds.
    alpha : float
        1/f spectral exponent.
    amplitude : float
        Noise amplitude.

    Returns
    -------
    dict
        1/f noise infidelity breakdown.
    """
    total_duration = gate_duration_h_s + (n_qubits - 1) * gate_duration_cnot_s
    per_qubit_dephasing = one_over_f_dephasing_rate(
        total_duration, alpha, amplitude
    )

    return {
        "protocol_duration_us": total_duration * 1e6,
        "dephasing_per_qubit": per_qubit_dephasing,
        "total_1f_infidelity": n_qubits * per_qubit_dephasing,
        "alpha": alpha,
        "amplitude": amplitude,
    }
