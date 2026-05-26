"""
Hardware validation module for GHZ secret sharing.

Executes GHZ preparation and the HBB protocol on real IBM Quantum
hardware, then compares results with simulation predictions.

Compatible with IBM Quantum Runtime (qiskit-ibm-runtime >= 0.40).
"""

import numpy as np
import time
import warnings
import os
import json

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def build_ghz_hardware_circuit(n_qubits, measure=True):
    """
    Build a GHZ preparation circuit suitable for hardware execution.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    measure : bool
        Whether to add measurement gates.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(n_qubits, n_qubits if measure else 0,
                        name=f"GHZ_{n_qubits}")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc


def build_ghz_x_basis_circuit(n_qubits):
    """
    Build GHZ circuit with X-basis measurement on all qubits.

    Used for coherence verification.
    """
    qc = QuantumCircuit(n_qubits, n_qubits, name=f"GHZ_{n_qubits}_X")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Rotate to X basis
    for i in range(n_qubits):
        qc.h(i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def transpile_for_hardware(circuits, backend, optimization_level=1):
    """
    Transpile circuits for a specific backend.

    Parameters
    ----------
    circuits : list[QuantumCircuit] or QuantumCircuit
        Circuits to transpile.
    backend : Backend
        IBM Quantum backend.
    optimization_level : int
        Transpilation optimization (0-3).

    Returns
    -------
    list or QuantumCircuit
        Transpiled circuits.
    """
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
    )

    if isinstance(circuits, list):
        return [pm.run(qc) for qc in circuits]
    return pm.run(circuits)


def run_ghz_on_hardware(backend, n_qubits_list, n_shots=4096):
    """
    Run GHZ preparation on real hardware for multiple party counts.

    Parameters
    ----------
    backend : Backend
        IBM Quantum backend.
    n_qubits_list : list[int]
        List of qubit counts to test.
    n_shots : int
        Number of shots per circuit.

    Returns
    -------
    dict
        Results including counts and estimated fidelities.
    """
    results = {}

    for n in n_qubits_list:
        print(f"\n  Running n={n} GHZ on {backend.name}...")

        # Z-basis circuit
        qc_z = build_ghz_hardware_circuit(n, measure=True)
        # X-basis circuit
        qc_x = build_ghz_x_basis_circuit(n)

        # Transpile
        qc_z_t = transpile_for_hardware(qc_z, backend)
        qc_x_t = transpile_for_hardware(qc_x, backend)

        print(f"    Z-basis depth: {qc_z_t.depth()}, X-basis depth: {qc_x_t.depth()}")

        # Execute
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=backend)

        t_start = time.time()

        job_z = sampler.run([qc_z_t], shots=n_shots)
        result_z = job_z.result()
        # Access classical register (default name is 'c')
        counts_z = result_z[0].data.c.get_counts()

        job_x = sampler.run([qc_x_t], shots=n_shots)
        result_x = job_x.result()
        counts_x = result_x[0].data.c.get_counts()

        elapsed = time.time() - t_start

        # Estimate fidelity from counts
        fid_z = _ghz_fidelity_from_z_counts(counts_z, n)
        coherence = _ghz_coherence_from_x_counts(counts_x, n)
        fid_estimate = (fid_z + coherence) / 2

        print(f"    Z-basis fidelity: {fid_z:.4f}")
        print(f"    X-basis coherence: {coherence:.4f}")
        print(f"    Estimated fidelity: {fid_estimate:.4f}")
        print(f"    Execution time: {elapsed:.1f} s")

        results[n] = {
            "n_qubits": n,
            "counts_z": counts_z,
            "counts_x": counts_x,
            "fidelity_z": fid_z,
            "coherence_x": coherence,
            "fidelity_estimate": fid_estimate,
            "transpiled_depth_z": qc_z_t.depth(),
            "transpiled_depth_x": qc_x_t.depth(),
            "n_shots": n_shots,
            "execution_time": elapsed,
            "backend": backend.name,
        }

    return results


def _ghz_fidelity_from_z_counts(counts, n):
    """
    Estimate GHZ population fidelity from Z-basis measurement.

    F_Z = (P(|00...0>) + P(|11...1>))
    """
    total = sum(counts.values())
    all_zero = counts.get("0" * n, 0)
    all_one = counts.get("1" * n, 0)
    return (all_zero + all_one) / total


def _ghz_coherence_from_x_counts(counts, n):
    """
    Estimate GHZ coherence from X-basis measurement.

    For ideal GHZ, X-basis measurement gives only even-parity outcomes.
    Coherence = P(even parity) - P(odd parity)
    """
    total = sum(counts.values())
    even_count = 0
    odd_count = 0
    for bitstring, count in counts.items():
        parity = sum(int(b) for b in bitstring) % 2
        if parity == 0:
            even_count += count
        else:
            odd_count += count
    return (even_count - odd_count) / total


def compare_hardware_simulation(hw_results, sim_results):
    """
    Compare hardware and simulation results.

    Parameters
    ----------
    hw_results : dict
        Hardware results from run_ghz_on_hardware.
    sim_results : list[dict]
        Simulation results with n_parties and fidelity_dm.

    Returns
    -------
    list[dict]
        Comparison data.
    """
    sim_dict = {r["n_parties"]: r for r in sim_results}
    comparison = []

    for n, hw in hw_results.items():
        sim_fid = sim_dict.get(n, {}).get("fidelity_dm", None)
        gap = (hw["fidelity_estimate"] - sim_fid) if sim_fid else None

        entry = {
            "n_qubits": n,
            "hardware_fidelity": hw["fidelity_estimate"],
            "simulation_fidelity": sim_fid,
            "gap": gap,
            "hardware_backend": hw["backend"],
        }
        comparison.append(entry)

        status = ""
        if gap is not None:
            status = f"Gap = {gap:+.4f}"
        print(f"  n={n}: HW={hw['fidelity_estimate']:.4f}, "
              f"Sim={sim_fid:.4f if sim_fid else 'N/A'}, {status}")

    return comparison


def save_hardware_results(results, output_dir):
    """Save hardware results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON (full data minus non-serializable)
    json_data = {}
    for n, r in results.items():
        json_data[str(n)] = {
            k: v for k, v in r.items()
            if k not in ["counts_z", "counts_x"]
        }
        json_data[str(n)]["counts_z"] = dict(r["counts_z"])
        json_data[str(n)]["counts_x"] = dict(r["counts_x"])

    json_path = os.path.join(output_dir, "hardware_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # CSV summary
    import csv
    csv_path = os.path.join(output_dir, "hardware_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_qubits", "fidelity_z", "coherence_x",
            "fidelity_estimate", "transpiled_depth_z",
            "n_shots", "backend",
        ])
        writer.writeheader()
        for n in sorted(results.keys()):
            r = results[n]
            writer.writerow({
                "n_qubits": r["n_qubits"],
                "fidelity_z": round(r["fidelity_z"], 6),
                "coherence_x": round(r["coherence_x"], 6),
                "fidelity_estimate": round(r["fidelity_estimate"], 6),
                "transpiled_depth_z": r["transpiled_depth_z"],
                "n_shots": r["n_shots"],
                "backend": r["backend"],
            })
    print(f"  Saved: {csv_path}")
