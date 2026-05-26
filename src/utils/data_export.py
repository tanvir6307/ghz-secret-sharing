"""
Data export manager for GHZ secret sharing simulations.

Ensures all simulation data is exported as CSV files for figure
reproduction without re-running simulations.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


class GHZDataExporter:
    """
    Data export manager for GHZ secret sharing simulations.

    Parameters
    ----------
    base_dir : str
        Base directory for data export.
    """

    def __init__(self, base_dir="data/simulation_results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._exported_files = []

    def _ensure_dir(self, filepath):
        """Ensure directory exists for filepath."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _save_csv(self, df, filepath):
        """Save DataFrame to CSV and track."""
        self._ensure_dir(filepath)
        df.to_csv(filepath, index=False)
        self._exported_files.append(filepath)
        print(f"  Exported: {filepath}")
        return filepath

    def export_ghz_fidelity_trajectory(self, n_qubits, result_data,
                                        filename=None):
        """
        Export GHZ preparation fidelity trajectory.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        result_data : dict
            Output from realistic_ghz_preparation().
        filename : str, optional
            Output filename.
        """
        if filename is None:
            filename = f"ghz_fidelity_trajectory_n{n_qubits}.csv"

        gate_seq = result_data["gate_sequence"]
        fid_traj = result_data["fidelity_trajectory"]
        err_contrib = result_data["error_contributions"]
        time_pts = result_data["time_points_ns"]

        rows = []
        for i, gate in enumerate(gate_seq):
            rows.append({
                "step_number": i,
                "time_ns": time_pts[i] if i < len(time_pts) else None,
                "gate_type": gate["gate_type"],
                "qubits_involved": str(gate["qubits"]),
                "description": gate["description"],
                "fidelity_after": fid_traj[i] if i < len(fid_traj) else None,
                "fidelity_loss": err_contrib[i] if i < len(err_contrib) else None,
                "cumulative_error": (
                    1 - fid_traj[i] if i < len(fid_traj) else None
                ),
            })

        df = pd.DataFrame(rows)
        filepath = os.path.join(
            self.base_dir, "ghz_preparation", filename
        )
        return self._save_csv(df, filepath)

    def export_ghz_scaling_analysis(self, scaling_results,
                                     filename="fidelity_scaling_model_fit.csv"):
        """
        Export GHZ fidelity scaling analysis.

        Parameters
        ----------
        scaling_results : list[dict] or dict
            Scaling study results.  Accepts a list of per-n dicts with
            keys ``n_parties``, ``fidelity_mean``, ``fidelity_std``, and
            optionally ``fit_fidelity``.
        """
        if isinstance(scaling_results, list):
            df = pd.DataFrame(scaling_results)
            rename_map = {
                "fidelity_mean": "mean_ghz_fidelity",
                "fidelity_std": "std_fidelity",
                "fit_fidelity": "expected_theoretical",
            }
            df.rename(columns={k: v for k, v in rename_map.items()
                               if k in df.columns}, inplace=True)
            if "std_fidelity" in df.columns:
                df["ci_95_lower"] = df["mean_ghz_fidelity"] - 1.96 * df["std_fidelity"]
                df["ci_95_upper"] = df["mean_ghz_fidelity"] + 1.96 * df["std_fidelity"]
        else:
            # Legacy columnar-dict path
            fid = scaling_results.get("ghz_fidelity", scaling_results.get("fidelity_mean", []))
            std = scaling_results.get("ghz_fidelity_std", scaling_results.get("fidelity_std", []))
            df = pd.DataFrame({
                "n_parties": scaling_results["n_parties"],
                "mean_ghz_fidelity": fid,
                "std_fidelity": std,
                "ci_95_lower": [f - 1.96 * s for f, s in zip(fid, std)],
                "ci_95_upper": [f + 1.96 * s for f, s in zip(fid, std)],
            })
            for src, dst in [("expected_fidelity_model", "expected_theoretical"),
                             ("circuit_depth", "circuit_depth_cnots"),
                             ("total_time_us", "total_time_us"),
                             ("secret_success_rate", "secret_reconstruction_success_rate"),
                             ("secret_success_std", "secret_reconstruction_std")]:
                if src in scaling_results:
                    df[dst] = scaling_results[src]

        filepath = os.path.join(
            self.base_dir, "scaling_analysis", filename
        )
        return self._save_csv(df, filepath)

    def export_error_budget(self, n_qubits, error_budget, filename=None):
        """
        Export error budget for n-qubit GHZ.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        error_budget : dict
            Error budget from compute_error_budget().
        """
        if filename is None:
            filename = f"error_budget_n{n_qubits}.csv"

        rows = []
        for source, data in error_budget.items():
            if isinstance(data, dict):
                rows.append({
                    "error_source": source,
                    "contribution_percent": data.get("percent", 0),
                    "contribution_absolute": data.get("absolute", 0),
                    "phase": data.get("phase", ""),
                    "notes": data.get("notes", ""),
                })
            elif isinstance(data, (int, float)):
                # Flat dict from compute_ghz_error_budget: source -> float
                rows.append({
                    "error_source": source,
                    "contribution_percent": data * 100,
                    "contribution_absolute": data,
                    "phase": "",
                    "notes": "",
                })

        df = pd.DataFrame(rows)
        filepath = os.path.join(self.base_dir, "error_budget", filename)
        return self._save_csv(df, filepath)

    def export_secret_sharing_results(self, results,
                                       filename="hbb_protocol_results.csv"):
        """
        Export secret sharing protocol results.

        Parameters
        ----------
        results : dict or list[dict]
            Protocol results from run_multiple_trials().
        """
        if isinstance(results, dict):
            results = [results]

        rows = []
        for r in results:
            rows.append({
                "n_parties": r["n_parties"],
                "secret_bits": r.get("secret_bits", ""),
                "mean_success_rate": r["mean_success_rate"],
                "std_success_rate": r["std_success_rate"],
                "mean_ghz_fidelity": r["mean_ghz_fidelity"],
                "std_ghz_fidelity": r.get("std_ghz_fidelity", 0),
                "mean_x_basis_qber": r.get("mean_x_basis_qber", ""),
                "std_x_basis_qber": r.get("std_x_basis_qber", ""),
                "perfect_reconstruction_rate": r.get(
                    "perfect_reconstruction_rate", 0
                ),
                "n_trials": r.get("n_trials", 1),
            })

        df = pd.DataFrame(rows)
        filepath = os.path.join(
            self.base_dir, "secret_sharing", filename
        )
        return self._save_csv(df, filepath)

    def export_security_analysis(self, security_results,
                                  filename="security_analysis.csv"):
        """
        Export security analysis results.

        Parameters
        ----------
        security_results : dict
            Output from full_security_analysis().
        """
        rows = [
            {
                "scenario": "honest",
                "success_rate": security_results["honest"]["success_rate"],
                "std": security_results["honest"]["std"],
                "degradation": 0,
                "x_basis_qber": security_results["honest"].get(
                    "x_basis_qber", ""
                ),
                "detectable": "N/A",
            },
            {
                "scenario": "intercept_resend",
                "success_rate": security_results["intercept_resend"][
                    "success_rate"
                ],
                "std": security_results["intercept_resend"]["std"],
                "degradation": security_results["ir_degradation"],
                "x_basis_qber": security_results["intercept_resend"].get(
                    "x_basis_qber", ""
                ),
                "detectable": security_results["ir_detectable"],
            },
            {
                "scenario": "entangle_measure",
                "success_rate": security_results["entangle_measure"][
                    "success_rate"
                ],
                "std": security_results["entangle_measure"]["std"],
                "degradation": security_results["em_degradation"],
                "x_basis_qber": security_results["entangle_measure"].get(
                    "x_basis_qber", ""
                ),
                "detectable": security_results["em_detectable"],
            },
        ]

        df = pd.DataFrame(rows)
        filepath = os.path.join(
            self.base_dir, "security_analysis", filename
        )
        return self._save_csv(df, filepath)

    def export_fidelity_threshold(self, threshold_results,
                                   filename="fidelity_security_threshold.csv"):
        """
        Export fidelity threshold analysis.

        Parameters
        ----------
        threshold_results : dict
            Output from fidelity_threshold_analysis().
        """
        df = pd.DataFrame({
            "noise_scale": threshold_results["noise_scales"],
            "ghz_fidelity": threshold_results["fidelities"],
            "success_rate": threshold_results["success_rates"],
            "security_met": threshold_results["security_met"],
        })

        filepath = os.path.join(
            self.base_dir, "security_analysis", filename
        )
        return self._save_csv(df, filepath)

    def export_experimental_comparison(self, comparison_data,
                                        filename="all_experimental_benchmarks.csv"):
        """
        Export comparison with experimental GHZ benchmarks.

        Parameters
        ----------
        comparison_data : list[dict]
            List of comparison results.
        """
        df = pd.DataFrame(comparison_data)
        filepath = os.path.join(
            self.base_dir, "literature_comparison", filename
        )
        return self._save_csv(df, filepath)

    def export_monte_carlo_convergence(self, mc_data,
                                        filename="monte_carlo_convergence.csv"):
        """
        Export Monte Carlo convergence data.

        Parameters
        ----------
        mc_data : dict
            Monte Carlo convergence data.
        """
        df = pd.DataFrame(mc_data)
        filepath = os.path.join(
            self.base_dir, "statistical_analysis", filename
        )
        return self._save_csv(df, filepath)

    def export_scaling_model_parameters(self, fit_params,
                                         filename="scaling_model_fit.csv"):
        """
        Export fitted scaling model parameters.
        """
        rows = []
        for key, val in fit_params.items():
            rows.append({
                "parameter_name": key,
                "value": val,
            })

        df = pd.DataFrame(rows)
        filepath = os.path.join(
            self.base_dir, "scaling_analysis", filename
        )
        return self._save_csv(df, filepath)

    def export_metadata(self, params, n_qubits_range,
                         filename="simulation_parameters_ghz.json"):
        """
        Export simulation metadata.
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "device": params["device_name"],
            "n_qubits_range": n_qubits_range,
            "noise_sources": 8,
            "framework": "Qiskit",
            "simulation_method": "density_matrix + automatic",
        }

        filepath = os.path.join(
            self.base_dir, "..", "metadata", filename
        )
        self._ensure_dir(filepath)
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  Exported: {filepath}")
        return filepath

    def get_exported_files(self):
        """Return list of all exported files."""
        return self._exported_files

    def summary(self):
        """Print export summary."""
        print(f"\nData Export Summary")
        print(f"  Total files exported: {len(self._exported_files)}")
        for fp in self._exported_files:
            print(f"    {fp}")
