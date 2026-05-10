"""Matplotlib helpers for training curves and baseline bar charts"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training(history: List[Dict], eval_history: List[Dict],
                  output_path: Path) -> Optional[Path]:
    if not history:
        return None
    episodes = [row["episode"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(episodes, [row["score"] for row in history],
                    label="train score", alpha=0.6)
    if eval_history:
        axes[0, 0].plot([row["episode"] for row in eval_history],
                        [row["mean_score"] for row in eval_history],
                        marker="o", label="eval mean score")
    axes[0, 0].set_title("Score (top-row crossings)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(episodes, [row["loss"] for row in history], color="crimson")
    axes[0, 1].set_title("Q-learning MSE loss")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(episodes, [row["q_value"] for row in history],
                    color="tab:green", label="train mean Q")
    if eval_history:
        axes[1, 0].plot([row["episode"] for row in eval_history],
                        [row["mean_max_q"] for row in eval_history],
                        marker="o", color="darkgreen", label="eval mean max Q")
    axes[1, 0].set_title("Predicted Q-values")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(episodes, [row["epsilon"] for row in history], color="orange")
    axes[1, 1].set_title("Epsilon")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].grid(alpha=0.25)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_baselines(baselines: List[Dict], dqn_eval: Optional[Dict],
                   output_path: Path) -> Optional[Path]:
    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for entry in baselines:
        labels.append(entry["policy"])
        means.append(entry["mean_score"])
        stds.append(entry["std_score"])
    if dqn_eval is not None:
        labels.append("dqn_basic")
        means.append(dqn_eval["mean_score"])
        stds.append(dqn_eval["std_score"])
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["tab:gray", "tab:olive", "tab:purple", "tab:cyan", "tab:blue"]
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=palette[: len(labels)])
    ax.set_ylabel("Mean score (10 episodes)")
    ax.set_title("Baselines vs Basic DQN — difficulty 0")
    for bar, mean in zip(bars, means):
        ax.annotate(f"{mean:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
