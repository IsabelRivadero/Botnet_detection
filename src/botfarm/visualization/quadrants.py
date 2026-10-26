from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_quadrants(
    scored_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    thr_if: float = 0.8,
    thr_lof: float = 0.8,
) -> plt.Figure:
    """Genera la Matriz de Detección 2D en Rangos Percentiles (IF x LOF)."""
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=150)

    x = scored_df.get("pct_if", scored_df.get("score_if", []))
    y = scored_df.get("pct_lof", scored_df.get("score_lof", []))
    quadrants = scored_df.get("quadrant", ["IV_NORMAL"] * len(scored_df))

    quad_config = {
        "IV_NORMAL": ("#5cb85c", "o", 60, "Q IV: Humano Orgánico"),
        "III_LEJANA_GLOBAL": ("#9b59b6", "s", 80, "Q III: Anomalía Global"),
        "II_GRUPO_LOCAL": ("#f0ad4e", "^", 90, "Q II: Bot Infiltrado Local"),
        "I_EXTREMA": ("#d9534f", "X", 120, "Q I: Botnet Masiva Crítica"),
    }

    ax.axvline(x=thr_if, color="#555555", linestyle="--", linewidth=1.2, label="Umbral IF")
    ax.axhline(y=thr_lof, color="#555555", linestyle=":", linewidth=1.2, label="Umbral LOF")

    for q_key, (color, marker, size, label_name) in quad_config.items():
        mask = quadrants == q_key
        if mask.any():
            ax.scatter(
                x[mask],
                y[mask],
                c=color,
                marker=marker,
                s=size,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.6,
                label=f"{label_name} (n={mask.sum()})",
            )

    ax.set_title("Espacio Complementario de Detección [IF x LOF]", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Percentil de Aislamiento Global - Isolation Forest ($S_{IF}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Percentil de Densidad Local - Local Outlier Factor ($S_{LOF}$)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, bbox_inches="tight")
        plt.close(fig)

    return fig