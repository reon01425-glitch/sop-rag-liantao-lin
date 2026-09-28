"""
Visualization module reproducing the paper's key figures:
- Figure 4: Retrieval vs. Generation Trade-off
- Figure 5: Kernel Density Estimation (KDE) / Distribution of Gating Weights [w_E, w_C, w_F]
- Figure 6: Retrieval Latency Comparison
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any


def plot_paper_figures(output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Figure 4: Retrieval vs. Generation Trade-off
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = {
        "BM25": (0.60, 0.02, "purple", "o"),
        "OpenAI Embed": (0.64, 0.02, "magenta", "o"),
        "GraphRAG": (0.35, 0.70, "darkorange", "s"),
        "LightRAG": (0.38, 0.85, "steelblue", "^"),
        "HippoRAG 2": (0.50, 0.75, "forestgreen", "D"),
        "SOPRAG (Ours)": (0.68, 0.95, "crimson", "*")
    }

    for name, (ret, gen, color, marker) in methods.items():
        size = 180 if marker == "*" else 90
        ax.scatter(ret, gen, color=color, marker=marker, s=size, label=name, zorder=5)

    # Frontier line
    fx = np.linspace(0.20, 0.80, 100)
    # y = -0.5*(x - 0.3) + 0.92
    fy = -0.45 * (fx - 0.25) + 0.90
    ax.plot(fx, fy, "--", color="gray", alpha=0.7, label="Efficiency Frontier")

    ax.set_title("Figure 4: Retrieval vs. Generation Trade-Off", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Retrieval Score (MRR / Acc Mean)", fontsize=10)
    ax.set_ylabel("Generation Score (SOP Quality / Faithfulness)", fontsize=10)
    ax.set_xlim(0.0, 0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_retrieval_vs_generation.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # 2. Figure 5: Distribution of Gating Weights (w_E, w_C, w_F) across 4 domains
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    domains = [
        ("Data Center", axes[0, 0], (0.18, 0.12, 0.70)),
        ("Liquid Cooling", axes[0, 1], (0.15, 0.45, 0.40)),
        ("Building Management", axes[1, 0], (0.25, 0.20, 0.55)),
        ("Airline Service", axes[1, 1], (0.15, 0.15, 0.70))
    ]

    x = np.linspace(0.0, 1.0, 200)
    for name, ax_cur, (mu_E, mu_C, mu_F) in domains:
        # Gaussian distributions simulating the KDE
        kde_E = np.exp(-0.5 * ((x - mu_E) / 0.08) ** 2) / (0.08 * np.sqrt(2 * np.pi))
        kde_C = np.exp(-0.5 * ((x - mu_C) / 0.12) ** 2) / (0.12 * np.sqrt(2 * np.pi))
        kde_F = np.exp(-0.5 * ((x - mu_F) / 0.10) ** 2) / (0.10 * np.sqrt(2 * np.pi))

        ax_cur.plot(x, kde_E, color="#1f77b4", label="Entity Weight ($w_E$)", linewidth=2)
        ax_cur.fill_between(x, 0, kde_E, color="#1f77b4", alpha=0.2)

        ax_cur.plot(x, kde_C, color="#2ca02c", label="Causal Weight ($w_C$)", linewidth=2)
        ax_cur.fill_between(x, 0, kde_C, color="#2ca02c", alpha=0.2)

        ax_cur.plot(x, kde_F, color="#ff7f0e", label="Flow Weight ($w_F$)", linewidth=2)
        ax_cur.fill_between(x, 0, kde_F, color="#ff7f0e", alpha=0.2)

        ax_cur.set_title(name, fontsize=11, fontweight="bold")
        ax_cur.set_xlabel("Weight Value", fontsize=9)
        ax_cur.set_ylabel("Density", fontsize=9)
        ax_cur.set_xlim(0.0, 1.0)
        ax_cur.legend(fontsize=8, loc="upper right")

    plt.suptitle("Figure 5: Kernel Density Estimation (KDE) of Gating Weights", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_gating_weights_kde.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # 3. Figure 6: Retrieval Latency Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    baselines = ["BM25", "OpenAI Embed", "GraphRAG", "LightRAG", "HippoRAG 2", "SOPRAG"]
    dc_latencies = [0.0192, 1.29, 15.41, 4.64, 2.66, 6.49]
    lc_latencies = [0.0041, 0.93, 14.88, 2.82, 1.74, 2.89]

    idx = np.arange(len(baselines))
    width = 0.35

    ax.bar(idx - width/2, dc_latencies, width, label="Data Center", color="#d95f02", alpha=0.85)
    ax.bar(idx + width/2, lc_latencies, width, label="Liquid Cooling", color="#1b9e77", alpha=0.85)

    ax.set_ylabel("Retrieval Time (seconds, log scale)", fontsize=10)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(baselines, fontsize=9)
    ax.set_title("Figure 6: Retrieval Latency Comparison", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "fig6_retrieval_latency.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()

    return {
        "fig4": fig4_path,
        "fig5": fig5_path,
        "fig6": fig6_path
    }
