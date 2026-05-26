"""
Unit tests for GHZ Secret Sharing project.

Covers: noise models, GHZ circuits, fidelity, HBB protocol,
security analysis, error budget, and data export.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestNoiseModels(unittest.TestCase):
    """Tests for individual and composite noise models."""

    def setUp(self):
        from src.utils.device_parameters import load_device_parameters
        self.params = load_device_parameters("ibmq_manila")

    def _thermal_args(self, n):
        p = self.params
        return (
            n,
            p.get("T1_per_qubit", [p["T1"]] * 7),
            p.get("T2_per_qubit", [p["T2"]] * 7),
            p["gate_duration_h"],
            p["gate_duration_cnot"],
        )

    def test_markovian_infidelity_positive(self):
        from src.noise_models.markovian_noise import ghz_thermal_infidelity
        for n in [3, 5, 7]:
            result = ghz_thermal_infidelity(*self._thermal_args(n))
            inf = result["total"]
            self.assertGreater(inf, 0, f"n={n}: infidelity should be > 0")
            self.assertLess(inf, 1, f"n={n}: infidelity should be < 1")

    def test_markovian_scales_with_n(self):
        from src.noise_models.markovian_noise import ghz_thermal_infidelity
        inf3 = ghz_thermal_infidelity(*self._thermal_args(3))["total"]
        inf7 = ghz_thermal_infidelity(*self._thermal_args(7))["total"]
        self.assertGreater(inf7, inf3, "More qubits should give more infidelity")

    def test_crosstalk_infidelity(self):
        from src.noise_models.crosstalk_noise import ghz_crosstalk_infidelity
        p = self.params
        result = ghz_crosstalk_infidelity(
            5, p["zz_coupling"], p["gate_duration_cnot_ns"],
            coupling_map=p["coupling_map"],
        )
        self.assertGreaterEqual(result["total"], 0)
        self.assertLess(result["total"], 0.5)

    def test_leakage_infidelity(self):
        from src.noise_models.leakage_noise import ghz_leakage_infidelity
        p = self.params
        result = ghz_leakage_infidelity(5, p["leakage_cnot"], p["leakage_single"])
        self.assertGreaterEqual(result["total"], 0)
        self.assertLess(result["total"], 0.5)

    def test_spam_infidelity_scales(self):
        from src.noise_models.spam_errors import ghz_spam_infidelity
        p = self.params
        ro = p.get("readout_error_per_qubit", [p["readout_error_mean"]] * 7)
        inf3 = ghz_spam_infidelity(3, p["thermal_population"], ro)["total_spam"]
        inf7 = ghz_spam_infidelity(7, p["thermal_population"], ro)["total_spam"]
        self.assertGreater(inf7, inf3)

    def test_coherent_infidelity(self):
        from src.noise_models.coherent_errors import ghz_coherent_infidelity
        result = ghz_coherent_infidelity(5, self.params["systematic_over_rotation"])
        self.assertGreaterEqual(result["total_average"], 0)

    def test_one_over_f_infidelity(self):
        from src.noise_models.one_over_f_noise import ghz_1f_infidelity
        p = self.params
        result = ghz_1f_infidelity(5, p["gate_duration_h"], p["gate_duration_cnot"])
        self.assertGreaterEqual(result["total_1f_infidelity"], 0)

    def test_collective_dephasing(self):
        from src.noise_models.collective_dephasing import ghz_collective_dephasing_infidelity
        p = self.params
        t2 = p.get("T2_per_qubit", [p["T2"]] * 7)
        result = ghz_collective_dephasing_infidelity(
            5, t2, p["gate_duration_h"], p["gate_duration_cnot"]
        )
        self.assertGreaterEqual(result["ghz_enhanced"], 0)

    def test_composite_noise_build(self):
        from src.noise_models.composite_noise import MultiQubitGHZNoise
        noise = MultiQubitGHZNoise(self.params)
        noise_model = noise.build_noise_model(qubits_used=list(range(3)))
        self.assertIsNotNone(noise_model)

    def test_composite_error_budget(self):
        from src.noise_models.composite_noise import MultiQubitGHZNoise
        noise = MultiQubitGHZNoise(self.params)
        budget = noise.compute_error_budget(n_qubits=5)
        self.assertIn("TOTAL_infidelity_estimate", budget)
        self.assertGreater(budget["TOTAL_infidelity_estimate"]["absolute"], 0)

    def test_factory_functions(self):
        from src.noise_models.composite_noise import (
            build_markovian_only_noise,
            build_depolarizing_only_noise,
            build_full_noise,
        )
        nm1 = build_markovian_only_noise(self.params)
        nm2 = build_depolarizing_only_noise(self.params)
        nm3 = build_full_noise(self.params)
        # These return MultiQubitGHZNoise objects
        self.assertIsNotNone(nm1)
        self.assertIsNotNone(nm2)
        self.assertIsNotNone(nm3)
        # Build actual noise models
        m1 = nm1.build_noise_model(qubits_used=[0, 1, 2])
        m2 = nm2.build_noise_model(qubits_used=[0, 1, 2])
        m3 = nm3.build_noise_model(qubits_used=[0, 1, 2])
        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)
        self.assertIsNotNone(m3)


class TestGHZCircuit(unittest.TestCase):
    """Tests for GHZ circuit construction."""

    def test_build_ghz_circuit(self):
        from src.ghz.ghz_circuit import build_ghz_circuit
        for n in [3, 4, 5, 6, 7]:
            qc = build_ghz_circuit(n)
            self.assertEqual(qc.num_qubits, n)

    def test_ghz_with_measurement(self):
        from src.ghz.ghz_circuit import build_ghz_circuit_with_measurement
        qc = build_ghz_circuit_with_measurement(5)
        self.assertEqual(qc.num_qubits, 5)
        self.assertEqual(qc.num_clbits, 5)

    def test_ideal_ghz_statevector(self):
        from src.ghz.ghz_circuit import get_ideal_ghz_statevector
        sv = get_ideal_ghz_statevector(3)
        self.assertEqual(len(sv), 8)
        # |000> and |111> should have equal amplitude
        self.assertAlmostEqual(abs(sv[0]), 1/np.sqrt(2), places=10)
        self.assertAlmostEqual(abs(sv[7]), 1/np.sqrt(2), places=10)

    def test_ideal_ghz_density_matrix(self):
        from src.ghz.ghz_circuit import get_ideal_ghz_density_matrix
        dm = get_ideal_ghz_density_matrix(3)
        self.assertEqual(dm.shape, (8, 8))
        # Trace should be 1
        self.assertAlmostEqual(np.trace(dm).real, 1.0, places=10)

    def test_circuit_depth(self):
        from src.ghz.ghz_circuit import ghz_circuit_depth
        # Returns a dict with depth info
        info3 = ghz_circuit_depth(3)
        self.assertIn("circuit_depth", info3)
        self.assertEqual(info3["n_cnot"], 2)
        info5 = ghz_circuit_depth(5)
        self.assertEqual(info5["n_cnot"], 4)

    def test_invalid_n_raises(self):
        from src.ghz.ghz_circuit import build_ghz_circuit
        with self.assertRaises(ValueError):
            build_ghz_circuit(1)


class TestGHZPreparation(unittest.TestCase):
    """Tests for realistic GHZ preparation."""

    def setUp(self):
        from src.utils.device_parameters import load_device_parameters
        from src.noise_models.composite_noise import build_depolarizing_only_noise
        self.params = load_device_parameters("ibmq_manila")
        self.noise_obj = build_depolarizing_only_noise(self.params)

    def test_realistic_preparation(self):
        from src.ghz.ghz_preparation import realistic_ghz_preparation
        result = realistic_ghz_preparation(3, self.noise_obj, self.params)
        self.assertIn("final_fidelity", result)
        self.assertIn("fidelity_trajectory", result)
        self.assertGreater(result["final_fidelity"], 0.5)
        self.assertLessEqual(result["final_fidelity"], 1.0)

    def test_expected_fidelity_analytical(self):
        from src.ghz.ghz_preparation import calculate_expected_fidelity
        result = calculate_expected_fidelity(3, self.params)
        f = result["F_total"]
        self.assertGreater(f, 0.5)
        self.assertLessEqual(f, 1.0)

    def test_fidelity_decreases_with_n(self):
        from src.ghz.ghz_preparation import calculate_expected_fidelity
        f3 = calculate_expected_fidelity(3, self.params)["F_total"]
        f7 = calculate_expected_fidelity(7, self.params)["F_total"]
        self.assertGreater(f3, f7)

    def test_experimental_benchmarks_exist(self):
        from src.ghz.ghz_preparation import EXPERIMENTAL_BENCHMARKS
        self.assertGreater(len(EXPERIMENTAL_BENCHMARKS), 5)


class TestFidelity(unittest.TestCase):
    """Tests for fidelity computation."""

    def test_ghz_fidelity_ideal(self):
        from src.fidelity.state_fidelity import ghz_state_fidelity
        from src.ghz.ghz_circuit import get_ideal_ghz_density_matrix
        dm = get_ideal_ghz_density_matrix(3)
        f = ghz_state_fidelity(dm, 3)
        self.assertAlmostEqual(f, 1.0, places=10)

    def test_ghz_fidelity_from_counts(self):
        from src.fidelity.state_fidelity import ghz_fidelity_from_counts
        # Perfect GHZ counts
        counts = {"000": 500, "111": 500}
        result = ghz_fidelity_from_counts(counts, 3)
        self.assertAlmostEqual(result["ghz_population"], 1.0, places=5)

    def test_ghz_fidelity_from_noisy_counts(self):
        from src.fidelity.state_fidelity import ghz_fidelity_from_counts
        # Noisy counts
        counts = {"000": 400, "111": 400, "001": 100, "010": 100}
        result = ghz_fidelity_from_counts(counts, 3)
        self.assertAlmostEqual(result["ghz_population"], 0.8, places=3)

    def test_entanglement_witness(self):
        from src.fidelity.state_fidelity import entanglement_witness
        # Ideal GHZ Z-basis counts
        counts = {"000": 500, "111": 500}
        result = entanglement_witness(counts, 3)
        self.assertTrue(result["is_entangled"], "Ideal GHZ should be detected as entangled")


class TestHBBProtocol(unittest.TestCase):
    """Tests for the HBB secret sharing protocol."""

    def setUp(self):
        from src.utils.device_parameters import load_device_parameters
        from src.noise_models.composite_noise import build_depolarizing_only_noise
        self.params = load_device_parameters("ibmq_manila")
        self.noise_obj = build_depolarizing_only_noise(self.params)

    def test_protocol_creation(self):
        from src.secret_sharing.hbb_protocol import GHZSecretSharingProtocol
        protocol = GHZSecretSharingProtocol(
            n_parties=3,
            noise_model_obj=self.noise_obj,
            params=self.params,
            shots=1024,
        )
        self.assertEqual(protocol.n_parties, 3)

    def test_share_single_bit(self):
        from src.secret_sharing.hbb_protocol import GHZSecretSharingProtocol
        protocol = GHZSecretSharingProtocol(
            n_parties=3,
            noise_model_obj=self.noise_obj,
            params=self.params,
            shots=1024,
        )
        result = protocol.share_secret("1", verbose=False)
        self.assertIn("mean_success_rate", result)
        self.assertIn("secret_bits", result)
        self.assertEqual(result["secret_bits"], "1")

    def test_share_multi_bit(self):
        from src.secret_sharing.hbb_protocol import GHZSecretSharingProtocol
        protocol = GHZSecretSharingProtocol(
            n_parties=3,
            noise_model_obj=self.noise_obj,
            params=self.params,
            shots=1024,
        )
        result = protocol.share_secret("101", verbose=False)
        self.assertIn("per_bit_results", result)
        self.assertEqual(len(result["per_bit_results"]), 3)

    def test_multiple_trials(self):
        from src.secret_sharing.hbb_protocol import GHZSecretSharingProtocol
        protocol = GHZSecretSharingProtocol(
            n_parties=3,
            noise_model_obj=self.noise_obj,
            params=self.params,
            shots=512,
        )
        results = protocol.run_multiple_trials("1", n_trials=3)
        self.assertIn("mean_success_rate", results)
        self.assertIn("all_success_rates", results)
        self.assertEqual(len(results["all_success_rates"]), 3)


class TestSecurityAnalysis(unittest.TestCase):
    """Tests for security analysis."""

    def setUp(self):
        from src.utils.device_parameters import load_device_parameters
        from src.noise_models.composite_noise import build_depolarizing_only_noise
        from src.secret_sharing.hbb_protocol import GHZSecretSharingProtocol
        self.params = load_device_parameters("ibmq_manila")
        self.noise_obj = build_depolarizing_only_noise(self.params)
        self.protocol = GHZSecretSharingProtocol(
            n_parties=3,
            noise_model_obj=self.noise_obj,
            params=self.params,
            shots=512,
        )

    def test_security_analyzer_creation(self):
        from src.secret_sharing.security_analysis import SecurityAnalyzer
        analyzer = SecurityAnalyzer(
            protocol=self.protocol,
            noise_model_obj=self.noise_obj,
            params=self.params,
        )
        self.assertIsNotNone(analyzer)

    def test_honest_protocol(self):
        from src.secret_sharing.security_analysis import SecurityAnalyzer
        analyzer = SecurityAnalyzer(
            protocol=self.protocol,
            noise_model_obj=self.noise_obj,
            params=self.params,
        )
        result = analyzer.run_honest_protocol(n_trials=3)
        self.assertIn("success_rate", result)

    def test_intercept_resend(self):
        from src.secret_sharing.security_analysis import SecurityAnalyzer
        analyzer = SecurityAnalyzer(
            protocol=self.protocol,
            noise_model_obj=self.noise_obj,
            params=self.params,
        )
        result = analyzer.simulate_intercept_resend_attack(n_trials=3)
        self.assertIn("success_rate", result)


class TestErrorBudget(unittest.TestCase):
    """Tests for error budget utilities."""

    def setUp(self):
        from src.utils.device_parameters import load_device_parameters
        self.params = load_device_parameters("ibmq_manila")

    def test_compute_budget(self):
        from src.utils.error_budget import compute_ghz_error_budget
        budget = compute_ghz_error_budget(3, self.params)
        self.assertIn("total", budget)
        self.assertIn("CNOT_errors", budget)
        self.assertGreater(budget["total"], 0)

    def test_budget_scales(self):
        from src.utils.error_budget import compute_ghz_error_budget
        b3 = compute_ghz_error_budget(3, self.params)
        b7 = compute_ghz_error_budget(7, self.params)
        self.assertGreater(b7["total"], b3["total"])

    def test_format_budget(self):
        from src.utils.error_budget import compute_ghz_error_budget, format_error_budget
        budget = compute_ghz_error_budget(3, self.params)
        text = format_error_budget(budget)
        self.assertIn("CNOT_errors", text)
        self.assertIn("total", text)

    def test_compare_budgets(self):
        from src.utils.error_budget import compute_ghz_error_budget, compare_error_budgets
        budgets = {n: compute_ghz_error_budget(n, self.params) for n in [3, 5]}
        comparison = compare_error_budgets(budgets)
        self.assertIn("n_parties", comparison)
        self.assertEqual(comparison["n_parties"], [3, 5])

    def test_cumulative_decay(self):
        from src.utils.error_budget import cumulative_fidelity_decay
        steps = cumulative_fidelity_decay(5, self.params)
        self.assertEqual(len(steps), 5)  # H + 4 CNOTs
        fidelities = [s["fidelity"] for s in steps]
        for i in range(1, len(fidelities)):
            self.assertLessEqual(fidelities[i], fidelities[i-1])


class TestStatisticalTests(unittest.TestCase):
    """Tests for statistical utilities."""

    def test_z_test(self):
        from src.utils.statistical_tests import z_test_proportion
        result = z_test_proportion(0.9, 0.85, 1000)
        self.assertIn("z_score", result)
        self.assertIn("p_value", result)

    def test_bootstrap_ci(self):
        from src.utils.statistical_tests import bootstrap_confidence_interval
        data = np.random.normal(0.9, 0.02, 100)
        result = bootstrap_confidence_interval(data)
        ci_low, ci_high = result["ci_lower"], result["ci_upper"]
        self.assertLess(ci_low, ci_high)
        self.assertLess(ci_low, 0.95)
        self.assertGreater(ci_high, 0.85)

    def test_cohens_d(self):
        from src.utils.statistical_tests import cohens_d
        result = cohens_d(np.array([0.9, 0.91, 0.89]), np.array([0.8, 0.81, 0.79]))
        self.assertGreater(abs(result["d"]), 0)

    def test_monte_carlo_convergence(self):
        from src.utils.statistical_tests import monte_carlo_convergence
        # Provide a callable that returns fidelity given shots
        def fake_fidelity(shots=1024):
            return 0.85 + np.random.normal(0, 0.01)
        result = monte_carlo_convergence(fake_fidelity, n_samples_list=[100, 500], n_repeats=3)
        self.assertIsInstance(result, dict)
        self.assertIn("n_samples", result)
        self.assertEqual(len(result["n_samples"]), 2)


class TestDataExport(unittest.TestCase):
    """Tests for data export."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_exporter_creation(self):
        from src.utils.data_export import GHZDataExporter
        exporter = GHZDataExporter(self.tmpdir)
        self.assertIsNotNone(exporter)

    def test_export_error_budget(self):
        from src.utils.data_export import GHZDataExporter
        exporter = GHZDataExporter(self.tmpdir)
        budget = {"CNOT_errors": 0.02, "readout": 0.01, "total": 0.03}
        exporter.export_error_budget(3, budget)
        path = os.path.join(self.tmpdir, "error_budget", "error_budget_n3.csv")
        self.assertTrue(os.path.exists(path))

    def test_export_metadata(self):
        from src.utils.data_export import GHZDataExporter
        from src.utils.device_parameters import load_device_parameters
        exporter = GHZDataExporter(self.tmpdir)
        params = load_device_parameters("ibmq_manila")
        exporter.export_metadata(params, [3, 5, 7])
        # metadata is exported to base_dir/../metadata/
        path = os.path.join(self.tmpdir, "..", "metadata", "simulation_parameters_ghz.json")
        self.assertTrue(os.path.exists(path))


class TestDeviceParameters(unittest.TestCase):
    """Tests for device parameter loading."""

    def test_load_manila(self):
        from src.utils.device_parameters import load_device_parameters
        params = load_device_parameters("ibmq_manila")
        self.assertEqual(params["device_name"], "ibmq_manila")
        self.assertIn("T1", params)
        self.assertIn("T2", params)
        self.assertIn("cnot_error_mean", params)

    def test_load_nairobi(self):
        from src.utils.device_parameters import load_device_parameters
        params = load_device_parameters("ibm_nairobi")
        self.assertEqual(params["device_name"], "ibm_nairobi")

    def test_load_torino(self):
        from src.utils.device_parameters import load_device_parameters
        params = load_device_parameters("ibm_torino")
        self.assertEqual(params["device_name"], "ibm_torino")

    def test_invalid_device(self):
        from src.utils.device_parameters import load_device_parameters
        with self.assertRaises((KeyError, FileNotFoundError, ValueError)):
            load_device_parameters("nonexistent_device")

    def test_ghz_qubit_layout(self):
        from src.utils.device_parameters import get_ghz_qubit_layout, load_device_parameters
        params = load_device_parameters("ibmq_manila")
        layout = get_ghz_qubit_layout(params, 3)
        self.assertEqual(len(layout), 3)

    def test_ghz_cnot_pairs(self):
        from src.utils.device_parameters import get_ghz_cnot_pairs, load_device_parameters
        params = load_device_parameters("ibmq_manila")
        pairs = get_ghz_cnot_pairs(params, 4)
        self.assertEqual(len(pairs), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
