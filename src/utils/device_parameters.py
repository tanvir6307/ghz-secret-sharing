"""
Device parameters loader and manager for GHZ secret sharing.

Loads experimentally-justified device parameters from JSON and provides
them in a convenient flat format for noise model construction and
GHZ circuit generation.
"""

import json
import os
import numpy as np


def get_default_params_path():
    """Return the default parameter file path."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "parameters", "justified_parameters.json")


def load_device_parameters(device_name="ibmq_manila", params_path=None):
    """
    Load device parameters from the justified parameters JSON.

    Parameters
    ----------
    device_name : str
        Name of the device (key in JSON).
    params_path : str, optional
        Path to the JSON file. Uses default if None.

    Returns
    -------
    dict
        Flat dictionary of device parameters ready for simulation use.
    """
    if params_path is None:
        params_path = get_default_params_path()

    with open(params_path, "r") as f:
        all_params = json.load(f)

    if device_name not in all_params:
        available = list(all_params.keys())
        raise ValueError(
            f"Device '{device_name}' not found. Available: {available}"
        )

    raw = all_params[device_name]

    # Build a flat, convenient parameter dict
    params = {
        "device_name": device_name,
        "device_type": raw["device_type"],
        "num_qubits": raw["num_qubits"],
        "max_ghz_qubits": raw.get("max_ghz_qubits", raw["num_qubits"]),
        "connectivity": raw["connectivity"],
        "coupling_map": [tuple(pair) for pair in raw["coupling_map"]],

        # GHZ qubit layouts for different party counts
        "ghz_qubit_layout": {
            int(k): v for k, v in raw.get("ghz_qubit_layout", {}).items()
        },

        # Coherence times (seconds)
        "T1": raw["T1"]["mean"],
        "T1_per_qubit": raw["T1"]["values_per_qubit"],
        "T1_std": raw["T1"].get("std", 0),
        "T2": raw["T2"]["mean"],
        "T2_per_qubit": raw["T2"]["values_per_qubit"],
        "T2_std": raw["T2"].get("std", 0),

        # Gate errors (probabilities)
        "single_qubit_error": raw["single_qubit_gate_error"]["mean"],
        "single_qubit_error_per_qubit": raw["single_qubit_gate_error"]["values_per_qubit"],
        "cnot_error_mean": raw["cnot_gate_error"]["mean"],
        "cnot_errors": {
            k: v for k, v in raw["cnot_gate_error"].items()
            if k not in ("mean", "source")
        },

        # Readout
        "readout_error_per_qubit": raw["readout_error"]["per_qubit"],
        "readout_error_mean": raw["readout_error"]["mean"],

        # Gate durations (seconds)
        "gate_duration_h": raw["gate_durations"]["single_qubit_ns"] * 1e-9,
        "gate_duration_x": raw["gate_durations"]["single_qubit_ns"] * 1e-9,
        "gate_duration_z": raw["gate_durations"]["single_qubit_ns"] * 1e-9,
        "gate_duration_cnot": raw["gate_durations"]["cnot_ns"] * 1e-9,
        "measurement_duration": raw["gate_durations"]["measurement_ns"] * 1e-9,
        "gate_duration_single_ns": raw["gate_durations"]["single_qubit_ns"],
        "gate_duration_cnot_ns": raw["gate_durations"]["cnot_ns"],
        "measurement_duration_ns": raw["gate_durations"]["measurement_ns"],

        # ZZ coupling (Hz)
        "zz_coupling": {
            k: v for k, v in raw["zz_coupling_Hz"].items()
            if k != "source"
        },

        # Leakage
        "leakage_single": raw["leakage_rates"]["single_qubit"],
        "leakage_cnot": raw["leakage_rates"]["cnot"],

        # 1/f noise
        "one_over_f_alpha": raw["one_over_f_noise"]["alpha"],
        "one_over_f_amplitude": raw["one_over_f_noise"]["amplitude"],

        # Thermal / SPAM
        "thermal_population": raw["thermal_population"]["value"],

        # Classical communication
        "classical_delay": raw["classical_communication_delay_ns"] * 1e-9,
        "classical_delay_ns": raw["classical_communication_delay_ns"],

        # Coherent errors
        "systematic_over_rotation": raw["systematic_over_rotation"],
    }

    return params


def get_ghz_qubit_layout(params, n_parties):
    """
    Get the qubit layout for an n-party GHZ state on the given device.

    Parameters
    ----------
    params : dict
        Device parameters.
    n_parties : int
        Number of parties (qubits).

    Returns
    -------
    list[int]
        Qubit indices to use for GHZ state.
    """
    layouts = params.get("ghz_qubit_layout", {})
    if n_parties in layouts:
        return layouts[n_parties]
    # Default: linear layout starting from qubit 0
    if n_parties > params["max_ghz_qubits"]:
        raise ValueError(
            f"Cannot create {n_parties}-qubit GHZ on {params['device_name']} "
            f"(max {params['max_ghz_qubits']})"
        )
    return list(range(n_parties))


def get_ghz_cnot_pairs(params, n_parties):
    """
    Get the ordered CNOT pairs needed for GHZ preparation.

    For an n-qubit GHZ state: H on qubit 0, then CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1).
    Maps these logical operations to physical qubit pairs.

    Parameters
    ----------
    params : dict
        Device parameters.
    n_parties : int
        Number of parties.

    Returns
    -------
    list[tuple[int, int]]
        List of (control, target) qubit pairs.
    """
    layout = get_ghz_qubit_layout(params, n_parties)
    cnot_pairs = []
    for i in range(len(layout) - 1):
        cnot_pairs.append((layout[i], layout[i + 1]))
    return cnot_pairs


def get_qubit_T1(params, qubit):
    """Get T1 for a specific qubit."""
    if qubit < len(params["T1_per_qubit"]):
        return params["T1_per_qubit"][qubit]
    return params["T1"]


def get_qubit_T2(params, qubit):
    """Get T2 for a specific qubit."""
    if qubit < len(params["T2_per_qubit"]):
        return params["T2_per_qubit"][qubit]
    return params["T2"]


def get_cnot_error(params, control, target):
    """Get CNOT error for a specific qubit pair."""
    key = f"({control},{target})"
    return params["cnot_errors"].get(key, params["cnot_error_mean"])


def get_readout_error(params, qubit):
    """Get readout error for a specific qubit."""
    if qubit < len(params["readout_error_per_qubit"]):
        return params["readout_error_per_qubit"][qubit]
    return params["readout_error_mean"]


def get_ghz_protocol_duration(params, n_parties):
    """
    Calculate total GHZ preparation protocol duration.

    Duration = 1 H gate + (n-1) CNOT gates.

    Parameters
    ----------
    params : dict
        Device parameters.
    n_parties : int
        Number of parties.

    Returns
    -------
    float
        Total duration in seconds.
    """
    t_h = params["gate_duration_h"]
    t_cnot = params["gate_duration_cnot"]
    return t_h + (n_parties - 1) * t_cnot


def print_device_summary(params):
    """Print a formatted summary of device parameters."""
    print(f"Device: {params['device_name']}")
    print(f"  Type: {params['device_type']}")
    print(f"  Qubits: {params['num_qubits']} (max GHZ: {params['max_ghz_qubits']})")
    print(f"  Connectivity: {params['connectivity']}")
    print(f"  T1: {params['T1']*1e6:.0f} μs (mean)")
    print(f"  T2: {params['T2']*1e6:.0f} μs (mean)")
    print(f"  Single-qubit error: {params['single_qubit_error']:.5f}")
    print(f"  CNOT error: {params['cnot_error_mean']:.5f}")
    print(f"  Readout error: {params['readout_error_mean']:.4f}")
    print(f"  Gate durations: H={params['gate_duration_single_ns']}ns, "
          f"CNOT={params['gate_duration_cnot_ns']}ns")
    print(f"  Leakage: single={params['leakage_single']}, "
          f"CNOT={params['leakage_cnot']}")
