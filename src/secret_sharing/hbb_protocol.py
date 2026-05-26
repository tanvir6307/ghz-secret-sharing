"""
Hillery-Buzek-Berthiaume (HBB) Quantum Secret Sharing Protocol.

Implements the complete HBB protocol for quantum secret sharing using
GHZ states under realistic noise conditions:

1. GHZ state preparation (Alice)
2. Secret encoding (Alice applies operations on her qubit)
3. Qubit distribution (quantum channel noise)
4. Measurement by all parties
5. Secret reconstruction (classical post-processing)

The protocol allows Alice to share a classical secret with n-1 other
parties such that all parties must cooperate to reconstruct it.

References:
    - Hillery, Buzek, Berthiaume (1999) PRA 59, 1829
    - Karlsson, Koashi, Imoto (1999) PRA 59, 162
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from ..ghz.ghz_circuit import build_ghz_circuit
from ..ghz.ghz_preparation import realistic_ghz_preparation


class GHZSecretSharingProtocol:
    """
    Complete HBB quantum secret sharing protocol.

    Implements GHZ-based secret sharing with complementary-basis (X)
    verification rounds, following Hillery-Buzek-Berthiaume (1999).

    Z-basis rounds:  encode & reconstruct the secret bit.
    X-basis rounds:  verify GHZ coherence (eavesdropping detection).

    Parameters
    ----------
    n_parties : int
        Number of parties (including Alice). Must be >= 3.
    noise_model_obj : MultiQubitGHZNoise
        Noise model object.
    params : dict
        Device parameters.
    shots : int
        Number of measurement shots per trial.
    check_fraction : float
        Fraction of shots devoted to X-basis verification (0–1).
    """

    def __init__(self, n_parties, noise_model_obj, params, shots=8192,
                 check_fraction=0.2):
        if n_parties < 3:
            raise ValueError(
                f"Secret sharing requires at least 3 parties, got {n_parties}"
            )
        self.n_parties = n_parties
        self.noise_model_obj = noise_model_obj
        self.params = params
        self.shots = shots
        self.check_fraction = check_fraction

    def _run_x_basis_check(self, n_check_shots=None, extra_gates_fn=None):
        """
        Run X-basis verification rounds to detect eavesdropping.

        For an intact n-qubit GHZ state, X-basis measurement yields ONLY
        outcomes with even parity.  Any loss of coherence (e.g. from an
        entangle-measure attack) introduces odd-parity outcomes.

        Parameters
        ----------
        n_check_shots : int, optional
            Shots for check rounds.  Defaults to check_fraction * shots.
        extra_gates_fn : callable, optional
            ``fn(qc, n)`` called after GHZ prep to inject attack gates
            (used by SecurityAnalyzer).

        Returns
        -------
        dict
            ``x_basis_qber`` (parity-violation rate), counts, shots.
        """
        if n_check_shots is None:
            n_check_shots = max(int(self.shots * self.check_fraction), 256)
        n = self.n_parties

        qc = QuantumCircuit(n + (1 if extra_gates_fn else 0),
                            n)  # +1 ancilla only when needed

        # GHZ preparation
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.barrier()

        # Hook for attack gates (SecurityAnalyzer injects Eve's ops here)
        if extra_gates_fn is not None:
            extra_gates_fn(qc, n)

        # X-basis measurement: H before Z-measurement = X-measurement
        for i in range(n):
            qc.h(i)
        qc.measure(range(n), range(n))

        # Execute
        qubits_used = list(range(n))
        noise_model = self.noise_model_obj.build_noise_model(
            qubits_used=qubits_used
        )
        sim = AerSimulator(method="automatic", noise_model=noise_model)
        result = sim.run(qc, shots=n_check_shots).result()
        counts = result.get_counts()

        # Parity check: ideal GHZ → even parity only
        total = sum(counts.values())
        even_count = 0
        for outcome, count in counts.items():
            # Strip any ancilla bits — keep only rightmost n chars
            protocol_str = outcome[-n:]
            parity = sum(int(b) for b in protocol_str) % 2
            if parity == 0:
                even_count += count

        qber = 1.0 - even_count / total

        return {
            "x_basis_qber": qber,
            "n_check_shots": n_check_shots,
            "counts": counts,
        }

    def share_secret(self, secret_bits, verbose=True):
        """
        Execute the complete HBB secret sharing protocol.

        Parameters
        ----------
        secret_bits : str
            Binary string representing the secret (e.g., "101").
        verbose : bool
            Print progress information.

        Returns
        -------
        dict
            Protocol results including success rate, fidelities, counts.
        """
        n = self.n_parties
        k = len(secret_bits)

        if verbose:
            print(f"\n{'='*60}")
            print(f"HBB Secret Sharing Protocol")
            print(f"  Parties: {n}")
            print(f"  Secret: '{secret_bits}' ({k} bits)")
            print(f"  Shots: {self.shots}")
            print(f"{'='*60}")

        results = {
            "secret_bits": secret_bits,
            "n_parties": n,
            "k_bits": k,
            "per_bit_results": [],
        }

        all_success_rates = []
        all_ghz_fidelities = []

        # Process each secret bit independently
        for bit_idx, bit in enumerate(secret_bits):
            if verbose:
                print(f"\n--- Secret bit {bit_idx}: '{bit}' ---")

            bit_result = self._share_single_bit(
                int(bit), verbose=verbose
            )
            results["per_bit_results"].append(bit_result)
            all_success_rates.append(bit_result["success_rate"])
            all_ghz_fidelities.append(bit_result["ghz_fidelity"])

        # Overall statistics
        results["mean_success_rate"] = np.mean(all_success_rates)
        results["std_success_rate"] = np.std(all_success_rates)
        results["mean_ghz_fidelity"] = np.mean(all_ghz_fidelities)

        # X-basis verification rounds (eavesdropping detection)
        check_result = self._run_x_basis_check()
        results["x_basis_qber"] = check_result["x_basis_qber"]

        # Full secret reconstruction
        reconstructed = ""
        for bit_result in results["per_bit_results"]:
            reconstructed += str(bit_result["reconstructed_bit"])

        results["reconstructed"] = reconstructed
        results["success"] = reconstructed == secret_bits
        results["overall_success_rate"] = np.prod(all_success_rates)

        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULTS")
            print(f"  Original secret:  '{secret_bits}'")
            print(f"  Reconstructed:    '{reconstructed}'")
            print(f"  Match: {results['success']}")
            print(f"  Mean success rate: {results['mean_success_rate']:.2%}")
            print(f"  Mean GHZ fidelity: {results['mean_ghz_fidelity']:.4f}")
            print(f"  X-basis QBER:      {results['x_basis_qber']:.4f}")
            print(f"{'='*60}")

        return results

    def _share_single_bit(self, secret_bit, verbose=True):
        """
        Share a single secret bit using the HBB protocol.

        Parameters
        ----------
        secret_bit : int
            0 or 1.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Single-bit protocol results.
        """
        n = self.n_parties

        # Step 1: Build the protocol circuit
        qc = QuantumCircuit(n, n)

        # GHZ preparation: H on q0, then CNOT cascade
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.barrier()

        # Step 2: Secret encoding by Alice (qubit 0)
        if secret_bit == 1:
            qc.x(0)
        qc.barrier()

        # Step 3: Distribution phase (channel noise is in the noise model)
        # (Idle time for distribution is implicitly modeled)

        # Step 4: All parties measure in Z basis
        qc.measure(range(n), range(n))

        # Execute with noise
        qubits_used = list(range(n))
        noise_model = self.noise_model_obj.build_noise_model(
            qubits_used=qubits_used
        )

        sim = AerSimulator(method="automatic", noise_model=noise_model)
        result = sim.run(qc, shots=self.shots).result()
        counts = result.get_counts()

        # Step 5: Reconstruct secret bit
        reconstructed, success_rate = self._reconstruct_bit(
            counts, secret_bit, n
        )

        # Estimate GHZ fidelity from counts
        ghz_fidelity = self._estimate_ghz_fidelity(counts, n, secret_bit)

        if verbose:
            print(f"  Secret bit: {secret_bit}")
            print(f"  Reconstructed: {reconstructed}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  GHZ fidelity est: {ghz_fidelity:.4f}")

        return {
            "secret_bit": secret_bit,
            "reconstructed_bit": reconstructed,
            "success_rate": success_rate,
            "ghz_fidelity": ghz_fidelity,
            "counts": counts,
            "circuit": qc,
        }

    def _reconstruct_bit(self, counts, secret_bit, n):
        """
        Reconstruct secret bit from measurement outcomes.

        In the HBB protocol with Z-basis measurement on an n-qubit GHZ:
        - For secret_bit = 0: ideal GHZ gives |00…0⟩ or |11…1⟩
          Alice's bit equals every other party's bit.
        - For secret_bit = 1: X on Alice gives |10…0⟩ or |01…1⟩
          Alice's bit is flipped relative to every other party.

        Reconstruction per shot:
          secret = alice_bit ⊕ majority(other_parties_bits)

        This works for both even and odd n.
        """
        total_shots = sum(counts.values())
        correct_count = 0

        for outcome, count in counts.items():
            # Qiskit bit ordering: rightmost bit is qubit 0 (Alice)
            bits = [int(b) for b in outcome]   # left=MSB, right=q0

            alice_bit = bits[-1]               # qubit 0 = Alice
            other_bits = bits[:-1]             # qubits 1..n-1

            # Majority vote among the other parties
            ones = sum(other_bits)
            others_majority = 1 if ones > len(other_bits) / 2 else 0

            reconstructed_shot = alice_bit ^ others_majority

            if reconstructed_shot == secret_bit:
                correct_count += count

        success_rate = correct_count / total_shots

        # Overall reconstruction via majority vote across all shots
        # (re-tally using same logic)
        secret_1_votes = 0
        secret_0_votes = 0
        for outcome, count in counts.items():
            bits = [int(b) for b in outcome]
            alice_bit = bits[-1]
            other_bits = bits[:-1]
            ones = sum(other_bits)
            others_majority = 1 if ones > len(other_bits) / 2 else 0
            recon = alice_bit ^ others_majority
            if recon == 1:
                secret_1_votes += count
            else:
                secret_0_votes += count

        reconstructed = 1 if secret_1_votes > secret_0_votes else 0

        return reconstructed, success_rate

    def _estimate_ghz_fidelity(self, counts, n, secret_bit):
        """
        Estimate GHZ fidelity from measurement counts.

        For secret_bit = 0: expect |00...0> and |11...1>
        For secret_bit = 1: expect |10...0> and |01...1>
        """
        total = sum(counts.values())

        if secret_bit == 0:
            target_0 = "0" * n
            target_1 = "1" * n
        else:
            # After X on qubit 0 (rightmost in Qiskit)
            target_0 = "0" * (n - 1) + "1"
            target_1 = "1" * (n - 1) + "0"

        p_target = (counts.get(target_0, 0) + counts.get(target_1, 0)) / total
        return p_target

    def run_multiple_trials(self, secret_bits, n_trials=20, verbose=False):
        """
        Run the protocol multiple times to gather statistics.

        Parameters
        ----------
        secret_bits : str
            Secret to share.
        n_trials : int
            Number of independent trials.
        verbose : bool
            Print per-trial info.

        Returns
        -------
        dict
            Aggregated statistics over all trials.
        """
        success_rates = []
        ghz_fidelities = []
        x_basis_qbers = []
        perfect_reconstructions = 0

        for trial in range(n_trials):
            result = self.share_secret(secret_bits, verbose=verbose)
            success_rates.append(result["mean_success_rate"])
            ghz_fidelities.append(result["mean_ghz_fidelity"])
            x_basis_qbers.append(result["x_basis_qber"])
            if result["success"]:
                perfect_reconstructions += 1

        return {
            "n_trials": n_trials,
            "secret_bits": secret_bits,
            "n_parties": self.n_parties,
            "mean_success_rate": np.mean(success_rates),
            "std_success_rate": np.std(success_rates),
            "mean_ghz_fidelity": np.mean(ghz_fidelities),
            "std_ghz_fidelity": np.std(ghz_fidelities),
            "mean_x_basis_qber": np.mean(x_basis_qbers),
            "std_x_basis_qber": np.std(x_basis_qbers),
            "perfect_reconstruction_rate": perfect_reconstructions / n_trials,
            "all_success_rates": success_rates,
            "all_ghz_fidelities": ghz_fidelities,
            "all_x_basis_qbers": x_basis_qbers,
        }
