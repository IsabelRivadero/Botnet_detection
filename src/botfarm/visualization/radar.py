from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from botfarm.features import FEATURES


def plot_radar_explanation(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Genera un gráfico de radar para comparar firmas por cuadrante."""
    cols = features or [c for c in FEATURES if c in df.columns][:6]
    if not cols or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos para radar", ha="center", va="center")
        return fig

    df_norm = df[cols].copy()
    for c in cols:
        min_v = df_norm[c].min()
        max_v = df_norm[c].max()
        df_norm[c] = (
            (df_norm[c] - min_v) / (max_v - min_v + 1e-8)
            if max_v > min_v
            else 0.5
        )

    num_vars = len(cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=150)

    group_col = "quadrant" if "quadrant" in df.columns else None
    if group_col:
        color_map = {
            "I_EXTREMA": "#d9534f",
            "II_GRUPO_LOCAL": "#f0ad4e",
            "III_LEJANA_GLOBAL": "#9b59b6",
            "IV_NORMAL": "#5cb85c",
        }
        for group, group_df in df_norm.groupby(df[group_col]):
            values = group_df[cols].mean().values.tolist()
            values += values[:1]
            c = color_map.get(str(group), "#3498db")
            ax.plot(angles, values, linewidth=2, label=str(group), color=c)
            ax.fill(angles, values, alpha=0.15, color=c)
    else:
        values = df_norm[cols].mean().values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label="Promedio General")
        ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_title("Firma Multidimensional por Cuadrante (Radar Chart)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, bbox_inches="tight")
        plt.close(fig)

    return fig