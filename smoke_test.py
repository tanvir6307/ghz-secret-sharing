#!/usr/bin/env python3
"""Quick end-to-end smoke test of all major modules."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
import numpy as np

print("=== 1. Device Parameters ===")
from src.utils.device_parameters import load_device_parameters
params = load_device_parameters("ibmq_manila")
print(f"  Device: {params['device_name']}")
print(f"  T1={params['T1']*1e6:.1f}us, T2={params['T2']*1e6:.1f}us")

print("=== 2. Noise Model ===")
from src.noise_models.composite_noise import MultiQubitGHZNoise, build_full_noise
noise_obj = build_full_noise(params)
nm = noise_obj.build_noise_model(qubits_used=[0, 1, 2])
print(f"  Noise model built: {type(nm).__name__}")

print("=== 3. Error Budget (composite) ===")
budget = noise_obj.compute_error_budget(3)
total_est = budget["TOTAL_infidelity_estimate"]["absolute"]
print(f"  Total infidelity estimate: {total_est:.4f}")

print("=== 4. GHZ Preparation ===")
from src.ghz.ghz_preparation import realistic_ghz_preparation
prep = realistic_ghz_preparation(3, noise_obj, params, shots=1024)
print(f"  GHZ fidelity (n=3): {prep['final_fidelity']:.4f}")

print("=== 5. HBB Protocol ===")
from src.secret_sharing.hbb_protocol import GHZSecretSharingProtocol
protocol = GHZSecretSharingProtocol(
    n_parties=3, noise_model_obj=noise_obj, params=params, shots=1024
)
result = protocol.share_secret("1", verbose=False)
print(f"  Success rate: {result['mean_success_rate']:.4f}")

print("=== 6. Security Analysis ===")
from src.secret_sharing.security_analysis import SecurityAnalyzer
analyzer = SecurityAnalyzer(protocol=protocol, noise_model_obj=noise_obj, params=params)
honest = analyzer.run_honest_protocol(n_trials=5)
print(f"  Honest success: {honest['success_rate']:.4f}")
ir = analyzer.simulate_intercept_resend_attack(n_trials=5)
print(f"  Intercept-resend success: {ir['success_rate']:.4f}")

print("=== 7. Error Budget Utility ===")
from src.utils.error_budget import compute_ghz_error_budget, format_error_budget
b = compute_ghz_error_budget(3, params)
print(f"  Error budget total: {b['total']:.4f}")
print(format_error_budget(b))

print("=== 8. Statistical Tests ===")
from src.utils.statistical_tests import z_test_proportion, bootstrap_confidence_interval
z_res = z_test_proportion(0.9, 0.85, 1000)
print(f"  Z-test p-value: {z_res['p_value']:.6f}")
ci = bootstrap_confidence_interval(np.random.normal(0.9, 0.02, 100))
print(f"  Bootstrap CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

print("\n" + "=" * 50)
print("ALL SMOKE TESTS PASSED SUCCESSFULLY")
print("=" * 50)
