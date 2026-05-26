"""
GHZ-based Quantum Secret Sharing Simulation.

A comprehensive simulation framework for studying GHZ-state-based quantum
secret sharing (HBB protocol) under realistic noise conditions on
superconducting IBM Quantum hardware.

Modules
-------
ghz : GHZ state preparation and circuits
secret_sharing : HBB protocol implementation and security analysis
noise_models : 8+ noise channels adapted for multi-qubit GHZ states
fidelity : State and process fidelity computations
validation : Hardware validation and literature comparison
utils : Device parameters, data export, statistics, visualization
"""

__version__ = "1.0.0"
__author__ = "Tanvir Hassan et al."
