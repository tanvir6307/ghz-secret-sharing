"""
Security analysis for GHZ-based quantum secret sharing.

Investigates how realistic noise affects the security of the HBB protocol:
1. Can noise be distinguished from eavesdropping?
2. At what fidelity does security break down?
3. How many eavesdropped qubits are detectable?

Attack simulations:
- Intercept-resend attack
- Entangle-measure attack
- Noise-masked eavesdropping

References:
    - Hillery, Buzek, Berthiaume (1999) PRA
    - Lo, Chau (1999) Science
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error


class SecurityAnalyzer:
    """
    Analyze security of GHZ-based secret sharing under noise.

    Parameters
    ----------
    protocol : GHZSecretSharingProtocol
        The HBB protocol instance.
    noise_model_obj : MultiQubitGHZNoise
        Noise model object.
    params : dict
        Device parameters.
    """

    def __init__(self, protocol, noise_model_obj, params):
        self.protocol = protocol
        self.noise_model_obj = noise_model_obj
        self.params = params
        self.n_parties = protocol.n_parties

    def run_honest_protocol(self, secret="1", n_trials=10):
        """
        Run the honest (no attack) protocol.

        Returns
        -------
        dict
            Baseline success rate, fidelity, and X-basis QBER.
        """
        results = self.protocol.run_multiple_trials(
            secret, n_trials=n_trials, verbose=False
        )
        return {
            "success_rate": results["mean_success_rate"],
            "std": results["std_success_rate"],
            "ghz_fidelity": results["mean_ghz_fidelity"],
            "x_basis_qber": results["mean_x_basis_qber"],
            "type": "honest",
        }

    def simulate_intercept_resend_attack(self, target_qubit=1,
                                          secret="1", n_trials=10):
        """
        Simulate an intercept-resend attack on one qubit.

        Eve intercepts the qubit sent to party `target_qubit`, measures it
        in a random basis, and resends a new qubit in the measured state.
        This destroys the entanglement with that qubit.

        The effect is modeled as complete depolarization of the target qubit.

        Parameters
        ----------
        target_qubit : int
            Which party's qubit Eve intercepts (1 to n-1).
        secret : str
            Secret to share.
        n_trials : int
            Number of trials.

        Returns
        -------
        dict
            Attack results including X-basis QBER.
        """
        n = self.n_parties
        shots = self.protocol.shots

        success_rates = []
        for _ in range(n_trials):
            # Build circuit
            qc = QuantumCircuit(n, n)
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            qc.barrier()

            # Secret encoding (single bit)
            secret_bit = int(secret[0])
            if secret_bit == 1:
                qc.x(0)
            qc.barrier()

            # Eve's attack: intercept-resend = complete depolarization
            # Reset the target qubit and randomly prepare |0> or |1>
            qc.reset(target_qubit)
            if np.random.random() > 0.5:
                qc.x(target_qubit)

            qc.measure(range(n), range(n))

            # Build noise model with attack (add full depolarization)
            qubits_used = list(range(n))
            noise_model = self.noise_model_obj.build_noise_model(
                qubits_used=qubits_used
            )

            sim = AerSimulator(method="automatic", noise_model=noise_model)
            result = sim.run(qc, shots=shots).result()
            counts = result.get_counts()

            # Calculate success rate using alice ⊕ majority(others)
            total = sum(counts.values())
            correct = 0
            for outcome, count in counts.items():
                bits = [int(b) for b in outcome]
                alice_bit = bits[-1]  # rightmost = q0 = Alice
                other_bits = bits[:-1]
                ones = sum(other_bits)
                others_majority = 1 if ones > len(other_bits) / 2 else 0
                recon = alice_bit ^ others_majority
                if recon == secret_bit:
                    correct += count
            success_rates.append(correct / total)

        # X-basis QBER under IR attack
        def _ir_attack_gates(qc_check, n_parties):
            qc_check.reset(target_qubit)
            if np.random.random() > 0.5:
                qc_check.x(target_qubit)
            qc_check.barrier()

        ir_check = self.protocol._run_x_basis_check(
            extra_gates_fn=_ir_attack_gates
        )

        return {
            "success_rate": np.mean(success_rates),
            "std": np.std(success_rates),
            "target_qubit": target_qubit,
            "x_basis_qber": ir_check["x_basis_qber"],
            "type": "intercept_resend",
        }

    def simulate_entangle_measure_attack(self, target_qubit=1,
                                          secret="1", n_trials=10):
        """
        Simulate an entangle-and-measure attack.

        Eve introduces an ancilla qubit, entangles it with the target qubit
        via CNOT, then measures her ancilla. This partially collapses the
        target qubit's state.

        In Z-basis only, the CNOT attack is invisible (classical copy).
        The X-basis verification rounds detect the coherence loss caused
        by Eve's CNOT: the GHZ state collapses to a classical mixture,
        producing ~50% parity-violation (QBER) in the X basis.

        Parameters
        ----------
        target_qubit : int
            Which party's qubit Eve attacks.
        secret : str
            Secret to share.
        n_trials : int
            Number of trials.

        Returns
        -------
        dict
            Attack results including X-basis QBER.
        """
        n = self.n_parties
        shots = self.protocol.shots

        success_rates = []
        for _ in range(n_trials):
            # Circuit with ancilla for Eve
            qc = QuantumCircuit(n + 1, n)  # Extra qubit for Eve

            # GHZ preparation on qubits 0..n-1
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            qc.barrier()

            # Secret encoding (single bit)
            secret_bit = int(secret[0])
            if secret_bit == 1:
                qc.x(0)
            qc.barrier()

            # Eve's attack: CNOT from target to Eve's ancilla (qubit n)
            qc.cx(target_qubit, n)
            qc.barrier()

            # Measurement (only protocol qubits, not Eve's)
            qc.measure(range(n), range(n))

            # Build noise model
            qubits_used = list(range(n + 1))
            noise_model = self.noise_model_obj.build_noise_model(
                qubits_used=list(range(n))
            )

            sim = AerSimulator(method="automatic", noise_model=noise_model)
            result = sim.run(qc, shots=shots).result()
            counts = result.get_counts()

            # Calculate success rate using alice ⊕ majority(others)
            total = sum(counts.values())
            correct = 0
            for outcome, count in counts.items():
                # Only look at the n protocol bits (rightmost n chars)
                protocol_bits_str = outcome[-n:]
                bits = [int(b) for b in protocol_bits_str]
                alice_bit = bits[-1]  # rightmost = q0 = Alice
                other_bits = bits[:-1]
                ones = sum(other_bits)
                others_majority = 1 if ones > len(other_bits) / 2 else 0
                recon = alice_bit ^ others_majority
                if recon == secret_bit:
                    correct += count
            success_rates.append(correct / total)

        # X-basis QBER under EM attack — the key detection mechanism.
        # Eve's CNOT decoheres the GHZ state, causing ~50% QBER.
        def _em_attack_gates(qc_check, n_parties):
            qc_check.cx(target_qubit, n_parties)  # ancilla is qubit n
            qc_check.barrier()

        em_check = self.protocol._run_x_basis_check(
            extra_gates_fn=_em_attack_gates
        )

        return {
            "success_rate": np.mean(success_rates),
            "std": np.std(success_rates),
            "target_qubit": target_qubit,
            "x_basis_qber": em_check["x_basis_qber"],
            "type": "entangle_measure",
        }

    def full_security_analysis(self, secret="1", n_trials=50):
        """
        Run complete security analysis.

        Detection uses TWO complementary metrics:
        1. Success-rate degradation (Z-basis) — detects IR attack.
        2. X-basis QBER — detects EM attack (coherence loss).

        Returns
        -------
        dict
            Comprehensive security analysis results.
        """
        print("Running security analysis...")
        print("-" * 50)

        # Baseline
        print("  Honest protocol...")
        honest = self.run_honest_protocol(secret, n_trials)

        # Intercept-resend attack
        print("  Intercept-resend attack...")
        ir_attack = self.simulate_intercept_resend_attack(
            target_qubit=1, secret=secret, n_trials=n_trials
        )

        # Entangle-measure attack
        print("  Entangle-measure attack...")
        em_attack = self.simulate_entangle_measure_attack(
            target_qubit=1, secret=secret, n_trials=n_trials
        )

        # Analysis — Z-basis degradation
        ir_degradation = honest["success_rate"] - ir_attack["success_rate"]
        em_degradation = honest["success_rate"] - em_attack["success_rate"]

        # Detection via Z-basis degradation (traditional)
        ir_detectable_z = ir_degradation > 0.10
        em_detectable_z = em_degradation > 0.10

        # Detection via X-basis QBER (complementary-basis verification)
        # Honest QBER is small (noise only); attacks raise it significantly
        ir_qber_excess = ir_attack["x_basis_qber"] - honest["x_basis_qber"]
        em_qber_excess = em_attack["x_basis_qber"] - honest["x_basis_qber"]
        ir_detectable_x = ir_qber_excess > 0.05
        em_detectable_x = em_qber_excess > 0.05

        # Combined: attack is detectable if EITHER channel flags it
        ir_detectable = ir_detectable_z or ir_detectable_x
        em_detectable = em_detectable_z or em_detectable_x

        results = {
            "honest": honest,
            "intercept_resend": ir_attack,
            "entangle_measure": em_attack,
            "ir_degradation": ir_degradation,
            "em_degradation": em_degradation,
            "ir_detectable": ir_detectable,
            "em_detectable": em_detectable,
            "ir_qber_excess": ir_qber_excess,
            "em_qber_excess": em_qber_excess,
            "n_parties": self.n_parties,
            "secret": secret,
        }

        print(f"\n{'='*50}")
        print("Security Analysis Results")
        print(f"{'='*50}")
        print(f"  Honest success rate:      {honest['success_rate']:.2%}")
        print(f"  Honest X-basis QBER:      {honest['x_basis_qber']:.4f}")
        print(f"  Intercept-resend rate:    {ir_attack['success_rate']:.2%}")
        print(f"    Z-basis degradation:    {ir_degradation:.2%}")
        print(f"    X-basis QBER:           {ir_attack['x_basis_qber']:.4f}")
        print(f"    Detectable:             {'Yes' if ir_detectable else 'NO'}")
        print(f"  Entangle-measure rate:    {em_attack['success_rate']:.2%}")
        print(f"    Z-basis degradation:    {em_degradation:.2%}")
        print(f"    X-basis QBER:           {em_attack['x_basis_qber']:.4f}")
        print(f"    Detectable:             {'Yes' if em_detectable else 'NO'}")

        return results

    def fidelity_threshold_analysis(self, n_fidelity_points=20,
                                     n_trials_per_point=5):
        """
        Determine minimum GHZ fidelity for secure secret sharing.

        Sweeps GHZ fidelity by varying noise scale and finds the
        threshold below which security degrades.

        Parameters
        ----------
        n_fidelity_points : int
            Number of noise scale values to test.
        n_trials_per_point : int
            Trials per noise level.

        Returns
        -------
        dict
            Threshold analysis results.
        """
        # Vary noise scale to achieve different fidelities
        noise_scales = np.linspace(0.1, 5.0, n_fidelity_points)

        fidelities = []
        success_rates = []
        security_met = []

        from ..noise_models.composite_noise import MultiQubitGHZNoise

        for scale in noise_scales:
            # Create noise model at this scale
            scaled_noise = MultiQubitGHZNoise(
                self.params, noise_scale=scale
            )
            from ..secret_sharing.hbb_protocol import GHZSecretSharingProtocol

            protocol = GHZSecretSharingProtocol(
                self.n_parties, scaled_noise, self.params,
                shots=self.protocol.shots
            )

            trial_results = protocol.run_multiple_trials(
                "1", n_trials=n_trials_per_point, verbose=False
            )

            fidelities.append(trial_results["mean_ghz_fidelity"])
            success_rates.append(trial_results["mean_success_rate"])
            security_met.append(trial_results["mean_success_rate"] > 2 / 3)

        # Find threshold
        threshold_fidelity = None
        for i, met in enumerate(security_met):
            if met:
                threshold_fidelity = fidelities[i]
                break

        return {
            "noise_scales": noise_scales.tolist(),
            "fidelities": fidelities,
            "success_rates": success_rates,
            "security_met": security_met,
            "threshold_fidelity": threshold_fidelity,
            "security_threshold": 2 / 3,
        }
