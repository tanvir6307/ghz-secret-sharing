#!/usr/bin/env python3
"""
Run GHZ state preparation on real IBM Quantum hardware
and compare with simulation results.

Usage:
    python run_hardware_validation.py
"""

import sys
import os
import json
import time
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.validation.hardware_validation import (
    build_ghz_hardware_circuit,
    build_ghz_x_basis_circuit,
    transpile_for_hardware,
    _ghz_fidelity_from_z_counts,
    _ghz_coherence_from_x_counts,
    save_hardware_results,
)

# ── Configuration ────────────────────────────────────────────────────
BACKEND_NAME = "ibm_marrakesh"
N_QUBITS_LIST = [3, 5, 7]
N_SHOTS = 4096
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "hardware_validation")

# Simulation results for comparison (from our simulation run)
SIM_FIDELITIES = {
    3: 0.9512,
    4: 0.9177,
    5: 0.8853,
    6: 0.8706,
    7: 0.8562,
}


def main():
    print("=" * 70)
    print("GHZ HARDWARE VALIDATION — IBM Quantum")
    print("=" * 70)
    print(f"  Backend:   {BACKEND_NAME}")
    print(f"  Qubits:    {N_QUBITS_LIST}")
    print(f"  Shots:     {N_SHOTS}")
    print()

    # ── Connect to IBM Quantum ────────────────────────────────────────
    print("Connecting to IBM Quantum...")
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")
    print(f"  Status:  {'Operational' if backend.status().operational else 'OFFLINE'}")

    if not backend.status().operational:
        print("ERROR: Backend is not operational. Try another backend.")
        sys.exit(1)

    # Print calibration info
    props = backend.properties()
    if props:
        print("\n  Calibration snapshot (first 7 qubits):")
        for i in range(min(7, backend.num_qubits)):
            try:
                t1 = props.qubit_property(i, 'T1')
                t2 = props.qubit_property(i, 'T2')
                print(f"    Qubit {i}: T1={t1[0]*1e6:.1f} μs, T2={t2[0]*1e6:.1f} μs")
            except Exception:
                pass

    # ── Build & transpile circuits ────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING AND TRANSPILING CIRCUITS")
    print("=" * 70)

    all_circuits = []
    circuit_info = []

    for n in N_QUBITS_LIST:
        qc_z = build_ghz_hardware_circuit(n, measure=True)
        qc_x = build_ghz_x_basis_circuit(n)

        qc_z_t = transpile_for_hardware(qc_z, backend, optimization_level=1)
        qc_x_t = transpile_for_hardware(qc_x, backend, optimization_level=1)

        print(f"\n  n={n}: Z-basis depth={qc_z_t.depth()}, X-basis depth={qc_x_t.depth()}")

        all_circuits.append(qc_z_t)
        all_circuits.append(qc_x_t)
        circuit_info.append({
            "n": n,
            "z_idx": len(all_circuits) - 2,
            "x_idx": len(all_circuits) - 1,
            "z_depth": qc_z_t.depth(),
            "x_depth": qc_x_t.depth(),
        })

    # ── Submit all circuits in a single job ───────────────────────────
    print("\n" + "=" * 70)
    print("SUBMITTING JOB TO IBM QUANTUM")
    print("=" * 70)

    sampler = Sampler(mode=backend)
    print(f"\n  Submitting {len(all_circuits)} circuits in one job...")
    t_submit = time.time()

    job = sampler.run(all_circuits, shots=N_SHOTS)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Waiting for results...")

    # Poll for status
    while True:
        status = job.status()
        print(f"    Status: {status} (elapsed: {time.time() - t_submit:.0f}s)")
        if status in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(15)

    if str(status) not in ("DONE", "JobStatus.DONE"):
        print(f"\n  ERROR: Job failed with status {status}")
        try:
            err = job.error_message()
            print(f"  Error: {err}")
        except Exception:
            pass
        sys.exit(1)

    elapsed_total = time.time() - t_submit
    print(f"\n  Job completed in {elapsed_total:.1f}s")

    # ── Extract results ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXTRACTING RESULTS")
    print("=" * 70)

    result = job.result()
    hw_results = {}

    for info in circuit_info:
        n = info["n"]
        z_result = result[info["z_idx"]]
        x_result = result[info["x_idx"]]

        # Try both register names
        try:
            counts_z = z_result.data.c.get_counts()
        except AttributeError:
            counts_z = z_result.data.meas.get_counts()

        try:
            counts_x = x_result.data.c.get_counts()
        except AttributeError:
            counts_x = x_result.data.meas.get_counts()

        fid_z = _ghz_fidelity_from_z_counts(counts_z, n)
        coherence = _ghz_coherence_from_x_counts(counts_x, n)
        fid_estimate = (fid_z + coherence) / 2

        print(f"\n  n={n} qubits:")
        print(f"    Z-basis fidelity (population): {fid_z:.4f}")
        print(f"    X-basis coherence:             {coherence:.4f}")
        print(f"    Estimated state fidelity:      {fid_estimate:.4f}")
        print(f"    Top Z-basis outcomes: {dict(sorted(counts_z.items(), key=lambda x: -x[1])[:5])}")

        hw_results[n] = {
            "n_qubits": n,
            "counts_z": dict(counts_z),
            "counts_x": dict(counts_x),
            "fidelity_z": fid_z,
            "coherence_x": coherence,
            "fidelity_estimate": fid_estimate,
            "transpiled_depth_z": info["z_depth"],
            "transpiled_depth_x": info["x_depth"],
            "n_shots": N_SHOTS,
            "backend": backend.name,
            "job_id": job_id,
            "execution_time_s": elapsed_total,
        }

    # ── Compare with simulation ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("HARDWARE vs SIMULATION COMPARISON")
    print("=" * 70)

    comparison = []
    print(f"\n  {'n':>3}  {'HW Fidelity':>12}  {'Sim Fidelity':>13}  {'Gap':>8}  {'Note'}")
    print(f"  {'---':>3}  {'------------':>12}  {'-------------':>13}  {'--------':>8}  {'--------------------'}")

    for n in sorted(hw_results.keys()):
        hw_fid = hw_results[n]["fidelity_estimate"]
        sim_fid = SIM_FIDELITIES.get(n, None)
        gap = hw_fid - sim_fid if sim_fid else None

        note = ""
        if gap is not None:
            if abs(gap) < 0.05:
                note = "Good agreement"
            elif gap < -0.05:
                note = "HW lower (expected)"
            else:
                note = "HW higher (unusual)"

        print(f"  {n:>3}  {hw_fid:>12.4f}  {sim_fid:>13.4f}  {gap:>+8.4f}  {note}")

        comparison.append({
            "n_qubits": n,
            "hardware_fidelity": round(hw_fid, 6),
            "hardware_fid_z": round(hw_results[n]["fidelity_z"], 6),
            "hardware_coherence_x": round(hw_results[n]["coherence_x"], 6),
            "simulation_fidelity": sim_fid,
            "gap": round(gap, 6) if gap else None,
            "backend": backend.name,
            "job_id": job_id,
        })

    # ── Save all outputs ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Full hardware results JSON
    json_path = os.path.join(OUTPUT_DIR, "hardware_results.json")
    json_save = {}
    for n, r in hw_results.items():
        json_save[str(n)] = r
    with open(json_path, "w") as f:
        json.dump(json_save, f, indent=2)
    print(f"  Saved: {json_path}")

    # 2. Hardware results CSV
    csv_path = os.path.join(OUTPUT_DIR, "hardware_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_qubits", "fidelity_z", "coherence_x",
            "fidelity_estimate", "transpiled_depth_z",
            "n_shots", "backend", "job_id",
        ])
        writer.writeheader()
        for n in sorted(hw_results.keys()):
            r = hw_results[n]
            writer.writerow({
                "n_qubits": r["n_qubits"],
                "fidelity_z": round(r["fidelity_z"], 6),
                "coherence_x": round(r["coherence_x"], 6),
                "fidelity_estimate": round(r["fidelity_estimate"], 6),
                "transpiled_depth_z": r["transpiled_depth_z"],
                "n_shots": r["n_shots"],
                "backend": r["backend"],
                "job_id": r["job_id"],
            })
    print(f"  Saved: {csv_path}")

    # 3. Comparison CSV
    comp_path = os.path.join(OUTPUT_DIR, "hardware_vs_simulation.csv")
    with open(comp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_qubits", "hardware_fidelity", "hardware_fid_z",
            "hardware_coherence_x", "simulation_fidelity",
            "gap", "backend", "job_id",
        ])
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)
    print(f"  Saved: {comp_path}")

    # 4. Calibration metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": backend.name,
        "n_qubits_total": backend.num_qubits,
        "n_qubits_tested": N_QUBITS_LIST,
        "n_shots": N_SHOTS,
        "job_id": job_id,
        "execution_time_s": round(elapsed_total, 1),
    }
    if props:
        cal = {}
        for i in range(min(7, backend.num_qubits)):
            try:
                t1 = props.qubit_property(i, 'T1')
                t2 = props.qubit_property(i, 'T2')
                cal[str(i)] = {
                    "T1_us": round(t1[0] * 1e6, 2),
                    "T2_us": round(t2[0] * 1e6, 2),
                }
            except Exception:
                pass
        meta["calibration"] = cal
    meta_path = os.path.join(OUTPUT_DIR, "hardware_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    print("\n" + "=" * 70)
    print("HARDWARE VALIDATION COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Outputs:    {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
