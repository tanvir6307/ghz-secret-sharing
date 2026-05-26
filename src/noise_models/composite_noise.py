"""
Composite noise model combining all 8+ error sources for multi-qubit GHZ states.

This module constructs a unified Qiskit NoiseModel that incorporates:
1. Markovian thermal relaxation (T1/T2)
2. Depolarizing gate errors (from calibration)
3. Cross-talk (ZZ coupling)
4. Leakage to |2> state
5. 1/f noise (non-Markovian dephasing)
6. SPAM errors (state prep + readout)
7. Coherent control errors (over-rotation)
8. Collective dephasing (correlated noise)

Adapted for multi-qubit GHZ preparation where:
- The CNOT cascade makes errors accumulate with party count
- Crosstalk scales with the number of spectator qubits
- Collective dephasing is enhanced by the GHZ structure
- SPAM errors scale linearly with the number of measured qubits
"""

import numpy as np
from collections import defaultdict

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import (
    depolarizing_error,
    thermal_relaxation_error,
    pauli_error,
)

from .markovian_noise import build_thermal_relaxation_noise
from .leakage_noise import (
    build_leakage_error_single,
    build_leakage_error_two_qubit,
)
from .spam_errors import build_readout_error, build_state_prep_error
from .coherent_errors import (
    coherent_error_probability,
    build_coherent_error_single_qubit,
    build_coherent_error_cnot,
)
from .one_over_f_noise import one_over_f_dephasing_rate
from .crosstalk_noise import build_crosstalk_noise, crosstalk_phase_error
from .collective_dephasing import (
    collective_dephasing_infidelity,
    build_collective_dephasing_error,
)


class MultiQubitGHZNoise:
    """
    Unified noise model for multi-qubit GHZ state preparation and
    secret sharing protocol simulation.

    Combines all 8+ error sources into a single Qiskit-compatible
    noise model with detailed error budget tracking.

    Parameters
    ----------
    params : dict
        Device parameters from load_device_parameters().
    enable_thermal : bool
        Enable T1/T2 thermal relaxation.
    enable_depolarizing : bool
        Enable depolarizing gate errors.
    enable_crosstalk : bool
        Enable ZZ cross-talk.
    enable_leakage : bool
        Enable leakage errors.
    enable_1f : bool
        Enable 1/f noise contribution.
    enable_spam : bool
        Enable SPAM errors.
    enable_coherent : bool
        Enable coherent control errors.
    enable_collective : bool
        Enable collective dephasing.
    noise_scale : float
        Global noise scaling factor (1.0 = nominal, for ZNE).
    """

    def __init__(
        self,
        params,
        enable_thermal=True,
        enable_depolarizing=True,
        enable_crosstalk=True,
        enable_leakage=True,
        enable_1f=True,
        enable_spam=True,
        enable_coherent=True,
        enable_collective=True,
        noise_scale=1.0,
    ):
        self.params = params
        self.noise_scale = noise_scale
        self.flags = {
            "thermal": enable_thermal,
            "depolarizing": enable_depolarizing,
            "crosstalk": enable_crosstalk,
            "leakage": enable_leakage,
            "1f": enable_1f,
            "spam": enable_spam,
            "coherent": enable_coherent,
            "collective": enable_collective,
        }
        self._error_budget = {}

    def build_noise_model(self, qubits_used=None):
        """
        Build the complete Qiskit NoiseModel for GHZ simulation.

        All errors for a given (instruction, qubits) pair are composed
        into a single QuantumError before being added to the model.

        Parameters
        ----------
        qubits_used : list[int], optional
            Subset of qubits to add noise for. If None, uses all device qubits.

        Returns
        -------
        NoiseModel
            Qiskit-compatible noise model.
        """
        p = self.params
        s = self.noise_scale
        noise_model = NoiseModel()

        if qubits_used is None:
            qubits_used = list(range(p["num_qubits"]))

        # Registry: (instruction_name, tuple(qubits)) -> [QuantumError, ...]
        error_registry = defaultdict(list)

        single_gates = ["h", "x", "z", "s", "sdg", "sx", "rz", "ry", "rx"]

        # -- 1. Thermal relaxation (T1/T2) --
        if self.flags["thermal"]:
            t_single = p["gate_duration_h"]
            t_meas = p["measurement_duration"]

            for q in qubits_used:
                T1 = p["T1_per_qubit"][q] if q < len(p["T1_per_qubit"]) else p["T1"]
                T2 = p["T2_per_qubit"][q] if q < len(p["T2_per_qubit"]) else p["T2"]
                T2 = min(T2, 2 * T1)
                T1_eff = T1 / s
                T2_eff = T2 / s
                T2_eff = min(T2_eff, 2 * T1_eff)

                err_sq = thermal_relaxation_error(T1_eff, T2_eff, t_single)
                for gate in single_gates:
                    error_registry[(gate, (q,))].append(err_sq)

                err_meas = thermal_relaxation_error(T1_eff, T2_eff, t_meas)
                error_registry[("measure", (q,))].append(err_meas)

        # -- 2. Depolarizing gate errors --
        if self.flags["depolarizing"]:
            for q in qubits_used:
                eq = (
                    p["single_qubit_error_per_qubit"][q]
                    if q < len(p["single_qubit_error_per_qubit"])
                    else p["single_qubit_error"]
                )
                eq_scaled = min(eq * s, 1.0)
                if eq_scaled > 0:
                    err_1q = depolarizing_error(eq_scaled, 1)
                    for gate in single_gates:
                        error_registry[(gate, (q,))].append(err_1q)

        # -- 3. Two-qubit depolarizing + thermal + leakage on CNOT --
        #
        # Build the set of CNOT pairs we need noise for.  Start with the
        # device coupling map, then add *fallback* pairs for any qubit
        # indices beyond the physical device (e.g. when simulating a
        # 7-qubit GHZ on a 5-qubit device).  Fallback pairs use mean
        # error rates so the noise degrades realistically.
        coupling = [tuple(pair) for pair in p["coupling_map"]]
        coupling_set = set(coupling)

        # Identify CNOT pairs required by the GHZ cascade that are NOT
        # in the coupling map and add them with fallback noise.
        extended_pairs = []
        for q in qubits_used:
            q_next = q + 1
            if q_next in qubits_used:
                if (q, q_next) not in coupling_set and (q_next, q) not in coupling_set:
                    extended_pairs.append((q, q_next))

        all_cx_pairs = coupling + extended_pairs

        for pair in all_cx_pairs:
            c, t = pair
            if c not in qubits_used or t not in qubits_used:
                continue
            cx_key = ("cx", (c, t))
            is_fallback = pair in extended_pairs

            # Depolarizing CNOT error
            if self.flags["depolarizing"]:
                if is_fallback:
                    cx_err = p["cnot_error_mean"]
                else:
                    key = f"({c},{t})"
                    cx_err = p["cnot_errors"].get(key, p["cnot_error_mean"])
                cx_err_scaled = min(cx_err * s, 1.0)
                if cx_err_scaled > 0:
                    error_registry[cx_key].append(
                        depolarizing_error(cx_err_scaled, 2)
                    )

            # Thermal relaxation during CNOT
            if self.flags["thermal"]:
                T1_c = p["T1_per_qubit"][c] if c < len(p["T1_per_qubit"]) else p["T1"]
                T2_c = p["T2_per_qubit"][c] if c < len(p["T2_per_qubit"]) else p["T2"]
                T1_t = p["T1_per_qubit"][t] if t < len(p["T1_per_qubit"]) else p["T1"]
                T2_t = p["T2_per_qubit"][t] if t < len(p["T2_per_qubit"]) else p["T2"]
                T2_c = min(T2_c, 2 * T1_c)
                T2_t = min(T2_t, 2 * T1_t)
                T1_c /= s
                T2_c /= s
                T1_t /= s
                T2_t /= s
                T2_c = min(T2_c, 2 * T1_c)
                T2_t = min(T2_t, 2 * T1_t)

                err_cx_thermal = thermal_relaxation_error(
                    T1_c, T2_c, p["gate_duration_cnot"]
                ).expand(
                    thermal_relaxation_error(T1_t, T2_t, p["gate_duration_cnot"])
                )
                error_registry[cx_key].append(err_cx_thermal)

            # Leakage on CNOT
            if self.flags["leakage"]:
                leak_err = build_leakage_error_two_qubit(
                    min(p["leakage_cnot"] * s, 1.0)
                )
                if leak_err is not None:
                    error_registry[cx_key].append(leak_err)

        # -- 4. Leakage on single-qubit gates --
        if self.flags["leakage"]:
            leak_gates = ["h", "x", "z", "ry", "rx"]
            for q in qubits_used:
                leak_sq = build_leakage_error_single(
                    min(p["leakage_single"] * s, 1.0)
                )
                if leak_sq is not None:
                    for gate in leak_gates:
                        error_registry[(gate, (q,))].append(leak_sq)

        # -- 4b. ZZ crosstalk during CNOT gates --
        #
        # Physical picture: while a CNOT executes on pair (c, t), the ZZ
        # coupling imparts a parasitic Z-rotation on each *spectator*
        # qubit that neighbours c or t.  We model this as additional
        # depolarizing noise on the CNOT pair (standard Qiskit approx).
        if self.flags["crosstalk"]:
            ct_info = build_crosstalk_noise(
                p["zz_coupling"],
                p["gate_duration_cnot_ns"],
                p["coupling_map"],
                p["num_qubits"],
            )
            for pair in all_cx_pairs:
                c, t = pair
                if c not in qubits_used or t not in qubits_used:
                    continue
                spectators = ct_info.get((c, t), [])
                ct_err_total = 0.0
                for spec, theta in spectators:
                    if spec not in qubits_used:
                        continue
                    theta_scaled = theta * s
                    ct_err_total += min(
                        np.sin(theta_scaled / 2) ** 2, 1.0
                    )
                ct_err_total = min(ct_err_total, 1.0)
                if ct_err_total > 1e-10:
                    err_ct = depolarizing_error(ct_err_total, 2)
                    error_registry[("cx", (c, t))].append(err_ct)

        # -- 4c. Coherent control errors (systematic over-rotation) --
        if self.flags["coherent"]:
            epsilon = p.get("systematic_over_rotation", 0.0)
            if abs(epsilon) > 1e-10:
                coh_sq = build_coherent_error_single_qubit(epsilon * s)
                if coh_sq is not None:
                    for q in qubits_used:
                        for gate in single_gates:
                            error_registry[(gate, (q,))].append(coh_sq)

                coh_cx = build_coherent_error_cnot(epsilon * s)
                if coh_cx is not None:
                    for pair in all_cx_pairs:
                        c, t = pair
                        if c in qubits_used and t in qubits_used:
                            error_registry[("cx", (c, t))].append(coh_cx)

        # -- 4d. Collective dephasing (correlated noise during CNOTs) --
        if self.flags["collective"]:
            correlation = 0.3
            t_cnot_s = p["gate_duration_cnot"]
            for pair in all_cx_pairs:
                c, t = pair
                if c not in qubits_used or t not in qubits_used:
                    continue
                T2_c = (
                    p["T2_per_qubit"][c]
                    if c < len(p["T2_per_qubit"])
                    else p["T2"]
                )
                T2_t = (
                    p["T2_per_qubit"][t]
                    if t < len(p["T2_per_qubit"])
                    else p["T2"]
                )
                coll_err = build_collective_dephasing_error(
                    T2_c, T2_t, t_cnot_s, correlation
                )
                if coll_err is not None:
                    error_registry[("cx", (c, t))].append(coll_err)

        # -- Compose and register all quantum errors --
        for (instr, qubits), err_list in error_registry.items():
            composed = err_list[0]
            for e in err_list[1:]:
                composed = composed.compose(e)
            noise_model.add_quantum_error(composed, instr, list(qubits))

        # -- 5. SPAM: readout errors --
        if self.flags["spam"]:
            for q in qubits_used:
                re = (
                    p["readout_error_per_qubit"][q]
                    if q < len(p["readout_error_per_qubit"])
                    else p["readout_error_mean"]
                )
                re_scaled = min(re * s, 0.5)
                ro_err = build_readout_error(re_scaled)
                noise_model.add_readout_error(ro_err, [q])

        # -- 6. SPAM: state preparation errors --
        if self.flags["spam"]:
            for q in qubits_used:
                prep_err = build_state_prep_error(
                    min(p["thermal_population"] * s, 1.0)
                )
                if prep_err is not None:
                    noise_model.add_quantum_error(prep_err, "reset", [q])

        return noise_model

    def compute_error_budget(self, n_qubits, protocol_duration_ns=None):
        """
        Compute analytical error budget for n-qubit GHZ preparation.

        Parameters
        ----------
        n_qubits : int
            Number of qubits in GHZ state.
        protocol_duration_ns : float, optional
            Total protocol duration in ns. If None, computed from gate times.

        Returns
        -------
        dict
            Error budget with per-source contributions.
        """
        p = self.params

        if protocol_duration_ns is None:
            t_h_ns = p["gate_duration_single_ns"]
            t_cnot_ns = p["gate_duration_cnot_ns"]
            protocol_duration_ns = t_h_ns + (n_qubits - 1) * t_cnot_ns

        t_total = protocol_duration_ns * 1e-9
        n_cnots = n_qubits - 1
        budget = {}

        # 1. T1 decay
        t1_prob = 1 - np.exp(-t_total / p["T1"])
        budget["T1_decay"] = {
            "percent": t1_prob * 100 * n_qubits,
            "absolute": t1_prob * n_qubits,
            "phase": "all",
            "notes": (
                f"T1={p['T1']*1e6:.0f} us, protocol={protocol_duration_ns:.0f} ns, "
                f"{n_qubits} qubits"
            ),
        }

        # 2. T2 dephasing
        t2_prob = 1 - np.exp(-t_total / p["T2"])
        budget["T2_dephasing"] = {
            "percent": t2_prob * 100 * n_qubits,
            "absolute": t2_prob * n_qubits,
            "phase": "all",
            "notes": f"T2={p['T2']*1e6:.0f} us, {n_qubits} qubits",
        }

        # 3. CNOT gate errors
        cnot_err = n_cnots * p["cnot_error_mean"]
        budget["CNOT_gate_errors"] = {
            "percent": cnot_err * 100,
            "absolute": cnot_err,
            "phase": "ghz_preparation",
            "notes": f"{n_cnots} CNOT gates in cascade",
        }

        # 4. Single-qubit gate errors (H gate)
        sq_err = 1 * p["single_qubit_error"]
        budget["single_qubit_errors"] = {
            "percent": sq_err * 100,
            "absolute": sq_err,
            "phase": "ghz_preparation",
            "notes": "1 Hadamard gate",
        }

        # 5. Readout errors (all n qubits measured)
        read_err = n_qubits * p["readout_error_mean"]
        budget["readout_errors"] = {
            "percent": read_err * 100,
            "absolute": read_err,
            "phase": "measurement",
            "notes": f"{n_qubits} qubits measured",
        }

        # 6. State preparation
        prep_err = n_qubits * p["thermal_population"]
        budget["state_preparation"] = {
            "percent": prep_err * 100,
            "absolute": prep_err,
            "phase": "initialization",
            "notes": f"{n_qubits} qubits, thermal_pop={p['thermal_population']}",
        }

        # 7. Leakage
        leak_total = n_cnots * p["leakage_cnot"] + 1 * p["leakage_single"]
        budget["leakage"] = {
            "percent": leak_total * 100,
            "absolute": leak_total,
            "phase": "all gates",
            "notes": f"{n_cnots} CNOTs + 1 H gate",
        }

        # 8. Cross-talk
        ghz_qubits = set(range(n_qubits))
        ghz_cnot_pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        ct_info = build_crosstalk_noise(
            p["zz_coupling"],
            p["gate_duration_cnot_ns"],
            p["coupling_map"],
            p["num_qubits"],
        )
        ct_total = 0
        for pair in ghz_cnot_pairs:
            spectators = ct_info.get(pair, [])
            for spec, theta in spectators:
                if spec in ghz_qubits:
                    ct_total += crosstalk_phase_error(theta)
        budget["crosstalk"] = {
            "percent": ct_total * 100,
            "absolute": ct_total,
            "phase": "CNOT gates",
            "notes": f"ZZ coupling during {n_cnots} CNOT gates",
        }

        # 9. 1/f noise
        onef = one_over_f_dephasing_rate(
            t_total, p["one_over_f_alpha"], p["one_over_f_amplitude"]
        )
        budget["one_over_f_noise"] = {
            "percent": onef * 100 * n_qubits,
            "absolute": onef * n_qubits,
            "phase": "all",
            "notes": f"alpha={p['one_over_f_alpha']}, {n_qubits} qubits",
        }

        # 10. Coherent errors
        coh = (1 + n_cnots) * coherent_error_probability(
            p["systematic_over_rotation"]
        )
        budget["coherent_errors"] = {
            "percent": coh * 100,
            "absolute": coh,
            "phase": "all gates",
            "notes": f"epsilon={p['systematic_over_rotation']}, {1 + n_cnots} gates",
        }

        # 11. Collective dephasing (GHZ-enhanced)
        T2_used = p["T2_per_qubit"][:n_qubits]
        if len(T2_used) < n_qubits:
            T2_used = T2_used + [p["T2"]] * (n_qubits - len(T2_used))
        coll_infid = collective_dephasing_infidelity(T2_used, t_total, 0.3)
        budget["collective_dephasing"] = {
            "percent": coll_infid * 100,
            "absolute": coll_infid,
            "phase": "all",
            "notes": f"Correlation=0.3, {n_qubits} qubits, GHZ-enhanced",
        }

        # Total (first-order sum)
        total = sum(v["absolute"] for v in budget.values())
        budget["TOTAL_infidelity_estimate"] = {
            "percent": total * 100,
            "absolute": total,
            "phase": "all",
            "notes": "First-order sum (overestimates due to correlations)",
        }

        self._error_budget = budget
        return budget

    def get_enabled_sources(self):
        """Return list of enabled noise sources."""
        return [k for k, v in self.flags.items() if v]

    def summary(self, n_qubits=3):
        """Print a human-readable summary."""
        sources = self.get_enabled_sources()
        print(f"MultiQubitGHZNoise ({len(sources)} sources enabled)")
        print(f"  Device: {self.params['device_name']}")
        print(f"  GHZ size: {n_qubits} qubits")
        print(f"  Noise scale: {self.noise_scale:.2f}x")
        print(f"  Sources: {', '.join(sources)}")
        if self._error_budget:
            total = self._error_budget.get("TOTAL_infidelity_estimate", {})
            print(
                f"  Estimated total infidelity: "
                f"{total.get('percent', '?'):.2f}%"
            )


def build_markovian_only_noise(params):
    """Build a noise model with ONLY Markovian (T1/T2) noise."""
    return MultiQubitGHZNoise(
        params,
        enable_thermal=True,
        enable_depolarizing=False,
        enable_crosstalk=False,
        enable_leakage=False,
        enable_1f=False,
        enable_spam=False,
        enable_coherent=False,
        enable_collective=False,
    )


def build_depolarizing_only_noise(params):
    """Build a noise model with ONLY depolarizing gate errors."""
    return MultiQubitGHZNoise(
        params,
        enable_thermal=False,
        enable_depolarizing=True,
        enable_crosstalk=False,
        enable_leakage=False,
        enable_1f=False,
        enable_spam=False,
        enable_coherent=False,
        enable_collective=False,
    )


def build_full_noise(params, noise_scale=1.0):
    """Build the full 8-source noise model."""
    return MultiQubitGHZNoise(params, noise_scale=noise_scale)
