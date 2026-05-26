"""
Publication-quality visualization suite for GHZ secret sharing study.

Generates all figures from simulation data (or CSV files for reproducibility).
Follows the research plan Figure 1-6 specification.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from typing import Optional


# Set publication-quality defaults
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (8, 5),
    "font.family": "serif",
})


def _save_figure(fig, save_path):
    """Helper to save and close a figure."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Figure 1: GHZ fidelity vs party count (scaling)
# ============================================================
def plot_ghz_scaling(scaling_data, save_path=None):
    """
    Figure 1: GHZ state fidelity vs number of parties.

    Parameters
    ----------
    scaling_data : list[dict] or pd.DataFrame
        With columns: n_parties, fidelity_mean, fidelity_std,
        optionally: fit_fidelity (model prediction).
    """
    if isinstance(scaling_data, list):
        df = pd.DataFrame(scaling_data)
    else:
        df = scaling_data

    fig, ax = plt.subplots(figsize=(9, 6))

    # Simulation data with error bars
    ax.errorbar(
        df["n_parties"], df["fidelity_mean"],
        yerr=df.get("fidelity_std", None),
        fmt="o-", color="#2196F3", markersize=10, capsize=5,
        linewidth=2, label="Simulation", zorder=5,
    )

    # Model fit (if available)
    if "fit_fidelity" in df.columns:
        ax.plot(
            df["n_parties"], df["fit_fidelity"],
            "--", color="#FF5722", linewidth=2, label="Exponential fit",
            zorder=4,
        )

    ax.set_xlabel("Number of Parties (n)")
    ax.set_ylabel("GHZ State Fidelity")
    ax.set_title("GHZ State Fidelity Scaling\nUnder Realistic IBM Quantum Noise")
    y_min = max(0.0, min(df["fidelity_mean"]) - 0.05)
    y_max = min(1.02, max(df["fidelity_mean"]) + 0.04)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3)

    _save_figure(fig, save_path)


# ============================================================
# Figure 2: Error budget stacked bar chart
# ============================================================
def plot_error_budget_stacked(error_budgets, save_path=None):
    """
    Figure 2: Error budget breakdown by party count (stacked bar).

    Parameters
    ----------
    error_budgets : dict[int, dict]
        Maps n_parties -> {source: contribution}.
    """
    n_values = sorted(error_budgets.keys())
    sources = set()
    for b in error_budgets.values():
        sources.update(k for k in b.keys() if k != "total")
    sources = sorted(sources)

    colors = sns.color_palette("Set2", len(sources))

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(n_values))
    bottoms = np.zeros(len(n_values))

    for i, src in enumerate(sources):
        values = [error_budgets[n].get(src, 0) for n in n_values]
        ax.bar(x, values, bottom=bottoms, label=src.replace("_", " "),
               color=colors[i], edgecolor="white", linewidth=0.5)
        bottoms += values

    ax.set_xlabel("Number of Parties")
    ax.set_ylabel("Total Infidelity")
    ax.set_title("Error Budget Breakdown by Party Count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.legend(loc="upper left", fontsize=10, frameon=True,
              ncol=1, borderpad=0.4, labelspacing=0.3)
    ax.grid(axis="y", alpha=0.3)

    _save_figure(fig, save_path)


# ============================================================
# Figure 3: Error budget pie chart (single n)
# ============================================================
def plot_error_budget_pie(error_budget, n_parties=3, save_path=None):
    """
    Figure 3: Error budget horizontal bar chart for a single party count.

    Parameters
    ----------
    error_budget : dict
        {source: contribution} (without 'total').
    n_parties : int
        Number of parties (for title).
    """
    labels = []
    sizes = []
    for source, value in error_budget.items():
        if source == "total":
            continue
        if value > 1e-6:
            labels.append(source.replace("_", " "))
            sizes.append(value)

    # Sort ascending so largest bar is at top
    paired = sorted(zip(sizes, labels))
    sizes = [p[0] for p in paired]
    labels = [p[1] for p in paired]
    total = sum(sizes)

    colors = sns.color_palette("Set2", len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, sizes, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    # Annotate each bar with percentage
    for bar, sz in zip(bars, sizes):
        pct = 100 * sz / total if total > 0 else 0
        ax.text(bar.get_width() + total * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sz:.4f} ({pct:.1f}%)",
                va="center", fontsize=12)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Infidelity Contribution")
    ax.set_title(f"GHZ Error Budget ({n_parties}-Party)\n"
                 f"Total Infidelity: {total:.4f}",
                 fontsize=18, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(sizes) * 1.35)

    fig.tight_layout()
    _save_figure(fig, save_path)


# ============================================================
# Figure 4: Secret sharing success rate vs fidelity
# ============================================================
def plot_secret_sharing_success(results, save_path=None):
    """
    Figure 4: Protocol success rate vs GHZ fidelity.

    Parameters
    ----------
    results : list[dict] or pd.DataFrame
        With columns: fidelity, success_rate, n_parties.
    """
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results

    fig, ax = plt.subplots(figsize=(9, 6))

    for n in sorted(df["n_parties"].unique()):
        subset = df[df["n_parties"] == n].sort_values("fidelity")
        ax.plot(subset["fidelity"], subset["success_rate"],
                "o-", label=f"n={n}", markersize=8, linewidth=2)

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5,
               label="Random guessing (50%)")
    ax.axvline(x=2/3, color="red", linestyle="--", linewidth=1.5,
               alpha=0.6, label="Classical threshold")

    ax.set_xlabel("GHZ State Fidelity")
    ax.set_ylabel("Secret Sharing Success Rate")
    ax.set_title("HBB Protocol Success Rate vs GHZ Fidelity")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3)

    _save_figure(fig, save_path)


# ============================================================
# Figure 5: Security analysis — threshold plot
# ============================================================
def plot_security_threshold(threshold_data, save_path=None):
    """
    Figure 5: GHZ fidelity and reconstruction success versus noise scale.

    Parameters
    ----------
    threshold_data : list[dict] or pd.DataFrame
        Preferred columns: noise_scale, ghz_fidelity, success_rate.
        The older column name honest_fidelity is accepted as an alias for
        ghz_fidelity.
    """
    if isinstance(threshold_data, list):
        df = pd.DataFrame(threshold_data)
    else:
        df = threshold_data

    if "ghz_fidelity" not in df.columns and "honest_fidelity" in df.columns:
        df = df.rename(columns={"honest_fidelity": "ghz_fidelity"})
    required = {"noise_scale", "ghz_fidelity", "success_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "threshold_data missing required columns: "
            + ", ".join(sorted(missing))
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.axhline(y=2/3, color="gray", linestyle=":", linewidth=1.5,
               label="2/3 threshold")
    ax.plot(df["noise_scale"], df["ghz_fidelity"],
            "o-", color="#1F77B4", label="GHZ fidelity",
            markersize=8, linewidth=2)
    ax.plot(df["noise_scale"], df["success_rate"],
            "s-", color="#D62728", label="Reconstruction success",
            markersize=8, linewidth=2)

    if "security_met" in df.columns:
        nonviable = df[~df["security_met"].astype(bool)]
        if not nonviable.empty:
            xmin = nonviable["noise_scale"].min()
            xmax = df["noise_scale"].max()
            ax.axvspan(xmin, xmax, color="#D62728", alpha=0.08,
                       label="Below success criterion")

    ax.set_xlabel("Noise Scale Factor")
    ax.set_ylabel("Probability / Fidelity")
    ax.set_title("Noise Threshold for GHZ Secret Sharing")
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3)

    _save_figure(fig, save_path)


# ============================================================
# Figure 6: Experimental comparison
# ============================================================
def plot_experimental_comparison(sim_data, exp_data, save_path=None):
    """
    Figure 6: Simulation vs experimental benchmarks.

    Parameters
    ----------
    sim_data : dict
        {n: fidelity} simulation results.
    exp_data : list[dict]
        Experimental benchmarks with keys: name, n_qubits, fidelity.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulation line
    n_vals = sorted(sim_data.keys())
    sim_fids = [sim_data[n] for n in n_vals]
    ax.plot(n_vals, sim_fids, "D-", color="#2196F3", markersize=12,
            linewidth=2.5, label="This work (simulation)", zorder=5)

    # Experimental points
    markers = ["o", "s", "^", "v", "P", "*", "X", "h", "p"]
    colors = sns.color_palette("tab10", len(exp_data))
    for i, exp in enumerate(exp_data):
        ax.scatter(exp["n_qubits"], exp["fidelity"],
                   marker=markers[i % len(markers)],
                   color=colors[i], s=120, edgecolor="black",
                   linewidth=1, label=exp["name"], zorder=6)

    ax.axhline(y=2/3, color="gray", linestyle=":", linewidth=1.5,
               label="Classical threshold")

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("GHZ State Fidelity")
    ax.set_title("Simulation vs Experimental Benchmarks")
    ax.legend(loc="lower left", fontsize=11, frameon=True)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.05)

    _save_figure(fig, save_path)


# ============================================================
# Additional figures
# ============================================================
def plot_monte_carlo_convergence(convergence_data, save_path=None):
    """
    Monte Carlo convergence plot showing fidelity stabilization.

    Parameters
    ----------
    convergence_data : list[dict] or pd.DataFrame
        With columns: n_shots, fidelity_mean, fidelity_std.
    """
    if isinstance(convergence_data, list):
        df = pd.DataFrame(convergence_data)
    else:
        df = convergence_data

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Mean fidelity
    ax1.semilogx(df["n_shots"], df["fidelity_mean"], "o-",
                 color="#2196F3", markersize=8, linewidth=2)
    ax1.fill_between(
        df["n_shots"],
        df["fidelity_mean"] - df["fidelity_std"],
        df["fidelity_mean"] + df["fidelity_std"],
        alpha=0.2, color="#2196F3",
    )
    ax1.set_xlabel("Number of Shots")
    ax1.set_ylabel("Mean Fidelity")
    ax1.set_title("Monte Carlo Convergence")
    ax1.grid(alpha=0.3)

    # Right: Standard deviation
    ax2.loglog(df["n_shots"], df["fidelity_std"], "s-",
               color="#FF5722", markersize=8, linewidth=2)
    ax2.set_xlabel("Number of Shots")
    ax2.set_ylabel("Fidelity Std Dev")
    ax2.set_title("Statistical Uncertainty")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_cumulative_fidelity_decay(steps_data, save_path=None):
    """
    Plot cumulative fidelity after each gate in GHZ cascade.

    Parameters
    ----------
    steps_data : list[dict]
        With keys: step, gate, fidelity.
    """
    if isinstance(steps_data, list):
        df = pd.DataFrame(steps_data)
    else:
        df = steps_data

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["step"], df["fidelity"], "o-", color="#9C27B0",
            markersize=10, linewidth=2.5)

    for _, row in df.iterrows():
        ax.annotate(
            row["gate"], (row["step"], row["fidelity"]),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=10,
        )

    ax.set_xlabel("Gate Step")
    ax.set_ylabel("Cumulative Fidelity")
    ax.set_title("GHZ State Fidelity Decay Through Circuit")
    ax.set_ylim(0.8, 1.02)
    ax.grid(alpha=0.3)

    _save_figure(fig, save_path)


def plot_heatmap_t1_t2(heatmap_data, save_path=None):
    """
    Heatmap of fidelity as a function of T1 and T2.

    Parameters
    ----------
    heatmap_data : pd.DataFrame
        Pivot table with T1 as index, T2 as columns, fidelity as values.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.3, vmax=1.0, ax=ax,
        cbar_kws={"label": "GHZ Fidelity"},
    )

    ax.set_xlabel("T2 (μs)")
    ax.set_ylabel("T1 (μs)")
    ax.set_title("GHZ Fidelity vs Coherence Times")

    _save_figure(fig, save_path)


def plot_noise_model_comparison(comparison_data, save_path=None):
    """
    Bar chart comparing fidelity across noise model configurations.

    Parameters
    ----------
    comparison_data : list[dict]
        With keys: model, fidelity.
    """
    if isinstance(comparison_data, list):
        df = pd.DataFrame(comparison_data)
    else:
        df = comparison_data

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("viridis", len(df))
    bars = ax.bar(df["model"], df["fidelity"], color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, fid in zip(bars, df["fidelity"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{fid:.4f}", ha="center", va="bottom", fontsize=12)

    ax.set_xlabel("Noise Model")
    ax.set_ylabel("GHZ Fidelity")
    ax.set_title("Impact of Individual Noise Sources on GHZ Fidelity")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    _save_figure(fig, save_path)


def generate_all_figures(data_dir, output_dir):
    """
    Generate all figures from CSV data files.

    Parameters
    ----------
    data_dir : str
        Base data directory.
    output_dir : str
        Output directory for figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Scaling
    scaling_path = os.path.join(data_dir, "scaling_analysis", "ghz_scaling_analysis.csv")
    if os.path.exists(scaling_path):
        df = pd.read_csv(scaling_path)
        plot_ghz_scaling(df, os.path.join(output_dir, "fig1_ghz_scaling.pdf"))

    # Figure 2: Error budget stacked
    budget_path = os.path.join(data_dir, "error_budget", "error_budget.csv")
    if os.path.exists(budget_path):
        df = pd.read_csv(budget_path)
        budgets = {}
        for n in df["n_parties"].unique():
            row = df[df["n_parties"] == n].iloc[0]
            budget = {col: row[col] for col in df.columns
                      if col not in ["n_parties", "total"]}
            budgets[int(n)] = budget
        plot_error_budget_stacked(budgets, os.path.join(output_dir, "fig2_error_budget.pdf"))

    # Figure 4: Secret sharing success
    ss_path = os.path.join(data_dir, "secret_sharing", "secret_sharing_results.csv")
    if os.path.exists(ss_path):
        df = pd.read_csv(ss_path)
        plot_secret_sharing_success(df, os.path.join(output_dir, "fig4_secret_sharing.pdf"))

    # Figure 5: Security analysis
    sec_path = os.path.join(data_dir, "security_analysis", "fidelity_threshold.csv")
    if os.path.exists(sec_path):
        df = pd.read_csv(sec_path)
        plot_security_threshold(df, os.path.join(output_dir, "fig5_security.pdf"))

    # Monte Carlo convergence
    mc_path = os.path.join(data_dir, "statistical_analysis", "monte_carlo_convergence.csv")
    if os.path.exists(mc_path):
        df = pd.read_csv(mc_path)
        plot_monte_carlo_convergence(df, os.path.join(output_dir, "monte_carlo_convergence.pdf"))

    print(f"\nAll available figures generated in: {output_dir}")
