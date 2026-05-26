#!/usr/bin/env python3
"""
Generate publication-quality figure of IBM hardware GHZ validation results.

Produces: figures/fig_hardware_validation.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ── Data ─────────────────────────────────────────────────────────────

# IBM hardware (ibm_marrakesh, 4096 shots, job d6a62bpd1ejc73c8b6h0)
hw_n      = np.array([3,    5,    7])
hw_fid    = np.array([0.9609, 0.9176, 0.8484])
hw_fid_z  = np.array([0.9673, 0.9373, 0.8892])
hw_coh_x  = np.array([0.9546, 0.8979, 0.8076])
# Shot-noise error bars: sigma = sqrt(f*(1-f)/n_shots)
hw_err_z  = np.sqrt(hw_fid_z * (1 - hw_fid_z) / 4096)
hw_err_x  = np.sqrt(np.abs(hw_coh_x) * (1 - np.abs(hw_coh_x)) / 4096)
hw_err    = np.sqrt(hw_fid * (1 - hw_fid) / 4096)

# Classical threshold
F_CLASSICAL = 2/3

# ── Figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

x_pos = np.arange(len(hw_n))
bar_w = 0.22

# Bars
bars_z = ax.bar(x_pos - bar_w, hw_fid_z, bar_w, color="#4393c3",
                edgecolor="white", linewidth=0.8,
                yerr=hw_err_z, capsize=4, error_kw={"lw": 1.2, "color": "#333"},
                label=r"$F_Z$ (Z-basis population)", zorder=3)
bars_x = ax.bar(x_pos, hw_coh_x, bar_w, color="#f4a582",
                edgecolor="white", linewidth=0.8,
                yerr=hw_err_x, capsize=4, error_kw={"lw": 1.2, "color": "#333"},
                label=r"$C_X$ (X-basis coherence)", zorder=3)
bars_t = ax.bar(x_pos + bar_w, hw_fid, bar_w, color="#b2182b",
                edgecolor="white", linewidth=0.8,
                yerr=hw_err, capsize=4, error_kw={"lw": 1.2, "color": "#333"},
                label=r"$\hat{F} = (F_Z + C_X)/2$", zorder=3)

# Value labels on bars
for bars in [bars_z, bars_x, bars_t]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Number of GHZ parties $n$")
ax.set_ylabel("Fidelity / Coherence")
ax.set_title("GHZ State Fidelity on IBM Marrakesh (4096 shots)",
             fontweight="bold", fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(n) for n in hw_n])
ax.set_ylim(0.70, 1.06)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc", fontsize=10)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

# ── Save ─────────────────────────────────────────────────────────────
fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)

out_pdf = os.path.join(fig_dir, "fig_hardware_validation.pdf")
out_png = os.path.join(fig_dir, "fig_hardware_validation.png")

fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
plt.close(fig)

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
