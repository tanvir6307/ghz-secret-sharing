#!/usr/bin/env python3
"""
==========================================================================
GHZ-BASED QUANTUM SECRET SHARING — COMPLETE SIMULATION PIPELINE
==========================================================================

This script executes the full research program:
1. GHZ state preparation with fidelity tracking (n=3..7)
2. Error budget analysis for each party count
3. HBB secret sharing protocol execution
4. Security analysis (intercept-resend, entangle-measure)
5. Scaling analysis with exponential fit
6. Noise model comparison (individual channels vs composite)
7. Monte Carlo convergence analysis
8. Comparison with experimental benchmarks
9. Data export to CSV
10. Publication-quality figure generation

All results are exported as CSV files for reproducibility.
"""

import sys
import os
import time
import warnings
import numpy as np
from scipy.optimize import curve_fit

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*quantum error already exists.*")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.device_parameters import load_device_parameters, print_device_summary
from src.noise_models.composite_noise import (
    MultiQubitGHZNoise,
    build_markovian_only_noise,
    build_depolarizing_only_noise,
    build_full_noise,
)
from src.ghz.ghz_circuit import (
    build_ghz_circuit,
    build_ghz_circuit_with_measurement,
    get_ideal_ghz_density_matrix,
    ghz_circuit_depth,
)
from src.ghz.ghz_preparation import (
    realistic_ghz_preparation,
    calculate_expected_fidelity,
    validate_against_experiment,
    EXPERIMENTAL_BENCHMARKS,
)
from src.fidelity.state_fidelity import (
    ghz_state_fidelity,
    ghz_fidelity_from_counts,
    entanglement_witness,
    full_ghz_fidelity_estimate,
)
from src.secret_sharing.hbb_protocol import GHZSecretSharingProtocol
from src.secret_sharing.security_analysis import SecurityAnalyzer
from src.utils.error_budget import (
    compute_ghz_error_budget,
    format_error_budget,
    compare_error_budgets,
    cumulative_fidelity_decay,
)
from src.utils.data_export import GHZDataExporter
from src.utils.statistical_tests import (
    bootstrap_confidence_interval,
    z_test_proportion,
    monte_carlo_convergence,
    cohens_d,
)
from src.utils.visualization import (
    plot_ghz_scaling,
    plot_error_budget_stacked,
    plot_error_budget_pie,
    plot_secret_sharing_success,
    plot_security_threshold,
    plot_monte_carlo_convergence,
    plot_cumulative_fidelity_decay,
    plot_noise_model_comparison,
)


# ── Configuration ────────────────────────────────────────────────────
DEFAULT_DEVICE = "ibmq_manila"
N_SHOTS = 8192
N_PARTIES_RANGE = [3, 4, 5, 6, 7]
SECRET_BITS = "101"
N_TRIALS = 100
MC_SHOT_COUNTS = [100, 500, 1000, 2000, 4000, 8192, 16384, 32768]


def exponential_decay(n, A, alpha, C):
    """Model: F(n) = A * exp(-alpha * n) + C"""
    return A * np.exp(-alpha * n) + C


def run_ghz_preparation(params, n_shots=N_SHOTS):
    """Section 1: GHZ state preparation for each party count."""
    print("\n" + "=" * 70)
    print("SECTION 1: GHZ STATE PREPARATION (n = 3..7)")
    print("=" * 70)

    from qiskit_aer import AerSimulator

    results = []
    fidelity_trajectories = {}

    for n in N_PARTIES_RANGE:
        print(f"\n  -- n = {n} parties --")

        # Build noise model for this party count
        noise_obj = MultiQubitGHZNoise(params)
        noise_model = noise_obj.build_noise_model(qubits_used=list(range(n)))

        # Density matrix simulation with per-gate fidelity tracking
        prep_result = realistic_ghz_preparation(n, noise_obj, params, shots=n_shots)
        fidelity_trajectories[n] = prep_result  # Store full result for exporter

        # Also simulate with Aer for measurement statistics
        qc = build_ghz_circuit_with_measurement(n)
        backend = AerSimulator(noise_model=noise_model)
        job = backend.run(qc, shots=n_shots)
        counts = job.result().get_counts()

        # Compute fidelity from counts
        fid_counts_data = ghz_fidelity_from_counts(counts, n)
        fid_counts = fid_counts_data["ghz_population"]

        # Entanglement witness
        witness_val = prep_result["final_fidelity"] - 0.5

        depth_info = ghz_circuit_depth(n)
        circuit_depth = depth_info["circuit_depth"]

        print(f"    Density matrix fidelity:  {prep_result['final_fidelity']:.4f}")
        print(f"    Counts-based fidelity:    {fid_counts:.4f}")
        print(f"    Entanglement witness:     {witness_val:.4f} "
              f"({'entangled' if witness_val > 0 else 'SEPARABLE'})")
        print(f"    Circuit depth:            {circuit_depth}")

        results.append({
            "n_parties": n,
            "fidelity_dm": prep_result["final_fidelity"],
            "fidelity_counts": fid_counts,
            "witness": witness_val,
            "circuit_depth": circuit_depth,
        })

    return results, fidelity_trajectories


def run_error_budget_analysis(params):
    """Section 2: Error budget for each party count."""
    print("\n" + "=" * 70)
    print("SECTION 2: ERROR BUDGET ANALYSIS")
    print("=" * 70)

    budgets = {}
    for n in N_PARTIES_RANGE:
        budget = compute_ghz_error_budget(n, params)
        budgets[n] = budget
        print(format_error_budget(budget, title=f"n = {n} Parties"))

    return budgets


def run_secret_sharing(params, n_shots=N_SHOTS):
    """Section 3: HBB secret sharing protocol."""
    print("\n" + "=" * 70)
    print("SECTION 3: HBB SECRET SHARING PROTOCOL")
    print("=" * 70)

    all_results = []

    for n in N_PARTIES_RANGE:
        print(f"\n  -- n = {n} parties, secret = '{SECRET_BITS}' --")

        noise_obj = MultiQubitGHZNoise(params)

        protocol = GHZSecretSharingProtocol(
            n_parties=n,
            noise_model_obj=noise_obj,
            params=params,
            shots=n_shots,
        )

        trial_results = protocol.run_multiple_trials(
            secret_bits=SECRET_BITS,
            n_trials=N_TRIALS,
        )

        mean_success = trial_results["mean_success_rate"]
        std_success = trial_results["std_success_rate"]
        mean_fid = trial_results["mean_ghz_fidelity"]
        mean_qber = trial_results["mean_x_basis_qber"]

        # Bootstrap CI
        ci_result = bootstrap_confidence_interval(
            np.array(trial_results["all_success_rates"]), n_bootstrap=1000
        )
        ci_low = ci_result["ci_lower"]
        ci_high = ci_result["ci_upper"]

        print(f"    Mean success rate:  {mean_success:.4f} +/- {std_success:.4f}")
        print(f"    95% CI:             [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"    Mean GHZ fidelity:  {mean_fid:.4f}")
        print(f"    X-basis QBER:       {mean_qber:.4f}")

        all_results.append({
            "n_parties": n,
            "secret_bits": SECRET_BITS,
            "mean_success_rate": mean_success,
            "std_success_rate": std_success,
            "mean_ghz_fidelity": mean_fid,
            "std_ghz_fidelity": trial_results["std_ghz_fidelity"],
            "mean_x_basis_qber": mean_qber,
            "std_x_basis_qber": trial_results["std_x_basis_qber"],
            "perfect_reconstruction_rate": trial_results["perfect_reconstruction_rate"],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "success_rate": mean_success,
            "fidelity": mean_fid,
            "n_trials": N_TRIALS,
        })

    return all_results


def run_security_analysis(params, n_shots=N_SHOTS):
    """Section 4: Security analysis with attack simulations."""
    print("\n" + "=" * 70)
    print("SECTION 4: SECURITY ANALYSIS")
    print("=" * 70)

    n = 3  # Focus on 3-party case for detailed analysis
    noise_obj = MultiQubitGHZNoise(params)

    protocol = GHZSecretSharingProtocol(
        n_parties=n,
        noise_model_obj=noise_obj,
        params=params,
        shots=n_shots,
    )

    analyzer = SecurityAnalyzer(
        protocol=protocol,
        noise_model_obj=noise_obj,
        params=params,
    )

    # Full security analysis
    print("\n  Running full security analysis...")
    security_results = analyzer.full_security_analysis()

    print(f"\n  Honest success rate:      {security_results['honest']['success_rate']:.4f}")
    print(f"  Honest X-basis QBER:      {security_results['honest']['x_basis_qber']:.4f}")
    print(f"  Intercept-resend rate:    {security_results['intercept_resend']['success_rate']:.4f}")
    print(f"  Entangle-measure rate:    {security_results['entangle_measure']['success_rate']:.4f}")

    print(f"\n  IR degradation:  {security_results['ir_degradation']:.4f}")
    print(f"  IR X-basis QBER: {security_results['intercept_resend']['x_basis_qber']:.4f}")
    print(f"  EM degradation:  {security_results['em_degradation']:.4f}")
    print(f"  EM X-basis QBER: {security_results['entangle_measure']['x_basis_qber']:.4f}")

    # Threshold analysis
    print("\n  Running fidelity threshold analysis...")
    threshold_data = analyzer.fidelity_threshold_analysis(
        n_fidelity_points=15, n_trials_per_point=5
    )

    return security_results, threshold_data


def run_scaling_analysis(prep_results):
    """Section 5: Scaling analysis with exponential fit."""
    print("\n" + "=" * 70)
    print("SECTION 5: SCALING ANALYSIS + EXPONENTIAL FIT")
    print("=" * 70)

    n_vals = np.array([r["n_parties"] for r in prep_results])
    fid_vals = np.array([r["fidelity_dm"] for r in prep_results])

    # Fit: F(n) = A * exp(-alpha * n) + C
    try:
        popt, pcov = curve_fit(
            exponential_decay, n_vals, fid_vals,
            p0=[1.0, 0.1, 0.0],
            bounds=([0, 0, -0.5], [2, 5, 1.0]),
            maxfev=10000,
        )
        A, alpha, C = popt
        perr = np.sqrt(np.diag(pcov))

        fit_fidelities = exponential_decay(n_vals, *popt)
        residuals = fid_vals - fit_fidelities
        r_squared = 1 - np.sum(residuals**2) / np.sum((fid_vals - np.mean(fid_vals))**2)

        print(f"\n  Exponential fit: F(n) = {A:.4f} * exp(-{alpha:.4f} * n) + {C:.4f}")
        print(f"  R^2 = {r_squared:.6f}")
        print(f"  Per-party fidelity loss: {alpha:.4f} +/- {perr[1]:.4f}")

        for n_pred in [8, 10, 15, 20]:
            f_pred = exponential_decay(n_pred, *popt)
            status = "(above classical)" if f_pred > 2/3 else "(BELOW classical)"
            print(f"  Predicted F(n={n_pred}):  {f_pred:.4f} {status}")

        scaling_data = []
        for i, n in enumerate(n_vals):
            f = fid_vals[i]
            scaling_data.append({
                "n_parties": int(n),
                "fidelity_mean": f,
                "fidelity_std": np.sqrt(f * (1 - f) / N_SHOTS),
                "fit_fidelity": fit_fidelities[i],
            })

        model_params = {
            "A": A, "alpha": alpha, "C": C,
            "R_squared": r_squared,
            "A_err": perr[0], "alpha_err": perr[1], "C_err": perr[2],
        }

    except RuntimeError as e:
        print(f"\n  Exponential fit failed: {e}")
        scaling_data = [
            {"n_parties": int(n), "fidelity_mean": f, "fidelity_std": np.sqrt(f * (1 - f) / N_SHOTS)}
            for n, f in zip(n_vals, fid_vals)
        ]
        model_params = None

    return scaling_data, model_params


def run_noise_comparison(params, n=3, n_shots=N_SHOTS):
    """Section 6: Compare noise model configurations."""
    print("\n" + "=" * 70)
    print("SECTION 6: NOISE MODEL COMPARISON")
    print("=" * 70)

    from qiskit_aer import AerSimulator

    qc = build_ghz_circuit_with_measurement(n)
    qubits = list(range(n))

    configs = [
        ("Ideal (no noise)", None),
        ("Depolarizing only", build_depolarizing_only_noise(params).build_noise_model(qubits)),
        ("Markovian only", build_markovian_only_noise(params).build_noise_model(qubits)),
        ("Full composite", build_full_noise(params).build_noise_model(qubits)),
    ]

    comparison = []
    for name, noise_model in configs:
        if noise_model is None:
            backend = AerSimulator()
        else:
            backend = AerSimulator(noise_model=noise_model)

        job = backend.run(qc, shots=n_shots)
        counts = job.result().get_counts()
        fid_data = ghz_fidelity_from_counts(counts, n)
        fid = fid_data["ghz_population"]

        print(f"  {name:25s}  F = {fid:.4f}")
        comparison.append({"model": name, "fidelity": fid})

    return comparison


def run_monte_carlo_convergence(params, n=3):
    """Section 7: Monte Carlo convergence analysis."""
    print("\n" + "=" * 70)
    print("SECTION 7: MONTE CARLO CONVERGENCE")
    print("=" * 70)

    from qiskit_aer import AerSimulator

    noise_obj = MultiQubitGHZNoise(params)
    noise_model = noise_obj.build_noise_model(qubits_used=list(range(n)))
    qc = build_ghz_circuit_with_measurement(n)
    backend = AerSimulator(noise_model=noise_model)

    convergence_data = []
    for shots in MC_SHOT_COUNTS:
        fidelities = []
        n_repeat = 20
        for _ in range(n_repeat):
            job = backend.run(qc, shots=shots)
            counts = job.result().get_counts()
            fid_data = ghz_fidelity_from_counts(counts, n)
            fidelities.append(fid_data["ghz_population"])

        mean_f = np.mean(fidelities)
        std_f = np.std(fidelities)
        print(f"  shots = {shots:6d}:  F = {mean_f:.4f} +/- {std_f:.4f}")

        convergence_data.append({
            "n_shots": shots,
            "fidelity_mean": mean_f,
            "fidelity_std": std_f,
        })

    return convergence_data


def run_experimental_comparison(prep_results):
    """Section 8: Compare simulation with experimental benchmarks."""
    print("\n" + "=" * 70)
    print("SECTION 8: COMPARISON WITH EXPERIMENTAL BENCHMARKS")
    print("=" * 70)

    sim_data = {r["n_parties"]: r["fidelity_dm"] for r in prep_results}

    exp_data = []
    for key, bench in EXPERIMENTAL_BENCHMARKS.items():
        exp_data.append({
            "name": key,
            "n_qubits": bench["n_qubits"],
            "fidelity": bench["fidelity"],
        })
        n_q = bench["n_qubits"]
        if n_q in sim_data:
            gap = sim_data[n_q] - bench["fidelity"]
            print(f"  {key:30s}  n={n_q}  Exp={bench['fidelity']:.3f}  "
                  f"Sim={sim_data[n_q]:.4f}  Gap={gap:+.4f}")
        else:
            print(f"  {key:30s}  n={n_q}  Exp={bench['fidelity']:.3f}  (no sim)")

    return sim_data, exp_data


def main():
    t_start_total = time.time()

    print("=" * 70)
    print("GHZ-BASED QUANTUM SECRET SHARING UNDER REALISTIC NOISE")
    print("Simulation Study for IBM Quantum Hardware")
    print("=" * 70)

    # -- Setup --
    params = load_device_parameters(DEFAULT_DEVICE)
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    data_dir = os.path.join(PROJECT_ROOT, "data", "simulation_results")
    os.makedirs(fig_dir, exist_ok=True)
    exporter = GHZDataExporter(data_dir)

    print(f"\nDevice: {params['device_name']}")
    print(f"T1 = {params['T1']*1e6:.0f} us,  T2 = {params['T2']*1e6:.0f} us")
    print(f"CNOT error = {params['cnot_error_mean']*100:.2f}%")
    print(f"Readout error = {params['readout_error_mean']*100:.2f}%")
    print(f"Parties: {N_PARTIES_RANGE}")
    print(f"Shots: {N_SHOTS}")

    # Section 1: GHZ Preparation
    prep_results, fid_trajectories = run_ghz_preparation(params)
    for n, prep_data in fid_trajectories.items():
        exporter.export_ghz_fidelity_trajectory(n, prep_data)

    # Section 2: Error Budget
    budgets = run_error_budget_analysis(params)
    for n_q, budget in budgets.items():
        exporter.export_error_budget(n_q, budget)

    decay_steps = cumulative_fidelity_decay(5, params)
    plot_cumulative_fidelity_decay(
        decay_steps,
        save_path=os.path.join(fig_dir, "cumulative_fidelity_decay.pdf"),
    )

    # Section 3: Secret Sharing
    ss_results = run_secret_sharing(params)
    exporter.export_secret_sharing_results(ss_results)

    # Section 4: Security Analysis
    security_results, threshold_data = run_security_analysis(params)
    exporter.export_security_analysis(security_results)
    exporter.export_fidelity_threshold(threshold_data)

    # Section 5: Scaling Analysis
    scaling_data, model_params = run_scaling_analysis(prep_results)
    exporter.export_ghz_scaling_analysis(scaling_data)
    if model_params:
        exporter.export_scaling_model_parameters(model_params)

    plot_ghz_scaling(
        scaling_data,
        save_path=os.path.join(fig_dir, "fig1_ghz_scaling.pdf"),
    )

    # Section 6: Noise Comparison
    noise_comparison = run_noise_comparison(params)
    plot_noise_model_comparison(
        noise_comparison,
        save_path=os.path.join(fig_dir, "noise_model_comparison.pdf"),
    )

    # Section 7: Monte Carlo Convergence
    convergence = run_monte_carlo_convergence(params)
    exporter.export_monte_carlo_convergence(convergence)
    plot_monte_carlo_convergence(
        convergence,
        save_path=os.path.join(fig_dir, "monte_carlo_convergence.pdf"),
    )

    # Section 8: Experimental Comparison
    sim_data, exp_data = run_experimental_comparison(prep_results)
    exporter.export_experimental_comparison(exp_data)
    # (No figure — discussed as text in manuscript)

    # Error budget figures
    plot_error_budget_stacked(
        budgets,
        save_path=os.path.join(fig_dir, "fig2_error_budget.pdf"),
    )
    if 3 in budgets:
        plot_error_budget_pie(
            budgets[3], n_parties=3,
            save_path=os.path.join(fig_dir, "fig3_error_budget_pie.pdf"),
        )

    # Secret sharing figure
    plot_secret_sharing_success(
        ss_results,
        save_path=os.path.join(fig_dir, "fig4_secret_sharing.pdf"),
    )

    # Security threshold figure
    if threshold_data:
        td = []
        for i in range(len(threshold_data["noise_scales"])):
            td.append({
                "noise_scale": threshold_data["noise_scales"][i],
                "ghz_fidelity": threshold_data["fidelities"][i],
                "success_rate": threshold_data["success_rates"][i],
                "security_met": threshold_data["security_met"][i],
            })
        plot_security_threshold(
            td,
            save_path=os.path.join(fig_dir, "fig5_security_threshold.pdf"),
        )

    # Export metadata
    elapsed = time.time() - t_start_total
    metadata = {
        "device_name": DEFAULT_DEVICE,
        "n_shots": N_SHOTS,
        "n_parties_range": N_PARTIES_RANGE,
        "n_trials_secret_sharing": N_TRIALS,
        "secret_bits": SECRET_BITS,
        "elapsed_seconds": round(elapsed, 1),
    }
    if model_params:
        metadata["scaling_fit"] = model_params
    exporter.export_metadata(metadata, N_PARTIES_RANGE)

    # Summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\n  Total time:  {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"  Data saved:  {data_dir}")
    print(f"  Figures:     {fig_dir}")
    print(f"\n  Key results:")
    for r in prep_results:
        above = "[OK]" if r["fidelity_dm"] > 2/3 else "[!!]"
        print(f"    n={r['n_parties']}: F={r['fidelity_dm']:.4f} {above}")
    print(f"\n  Secret sharing (n=3): {ss_results[0]['success_rate']:.1%} success rate")
    print("=" * 70)


if __name__ == "__main__":
    main()
