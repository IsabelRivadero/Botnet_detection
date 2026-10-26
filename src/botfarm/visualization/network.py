from pathlib import Path
from typing import Optional
import math
import pandas as pd


def generate_network_html(
    scored_df: pd.DataFrame,
    output_path: str | Path = "reports/red_botnets_islas_grid.html",
) -> str:
    """Genera un mapa HTML interactivo de comunidades dispuestas en rejilla de islas (Grid Layout)."""
    if scored_df.empty:
        return ""

    nodes_js = []
    num_communities = len(scored_df)
    cols = max(1, int(math.ceil(math.sqrt(num_communities))))
    spacing = 600

    quad_colors = {
        "I_EXTREMA": ("#d9534f", "diamond", 25, "BOTNET MASIVA CRÍTICA"),
        "II_GRUPO_LOCAL": ("#f0ad4e", "triangle", 20, "BOT INFILTRADO LOCAL"),
        "III_LEJANA_GLOBAL": ("#9b59b6", "square", 20, "ANOMALÍA GLOBAL"),
        "IV_NORMAL": ("#5cb85c", "dot", 12, "HUMANO ORGÁNICO"),
    }

    for idx, (cid, row) in enumerate(scored_df.iterrows()):
        r_idx = idx // cols
        c_idx = idx % cols
        center_x = c_idx * spacing
        center_y = r_idx * spacing

        quadrant = str(row.get("quadrant", "IV_NORMAL"))
        color, shape, base_size, class_label = quad_colors.get(
            quadrant, ("#5cb85c", "dot", 12, "HUMANO ORGÁNICO")
        )

        size = int(row.get("size", 10))
        node_size = min(max(base_size, size * 2), 45)

        if_score = float(row.get("score_if", row.get("pct_if", 0.0)))
        lof_score = float(row.get("score_lof", row.get("pct_lof", 0.0)))

        hover_title = (
            f"<b>Comunidad:</b> #{cid}<br>"
            f"<b>Clasificación:</b> {class_label}<br>"
            f"<b>Cuadrante:</b> {quadrant}<br>"
            f"<b>Tamaño:</b> {size} miembros<br>"
            f"<b>Score IF:</b> {if_score:.2f} | <b>Score LOF:</b> {lof_score:.2f}"
        )

        nodes_js.append(
            f"{{id: 'C_{cid}', label: 'Comunidad {cid}', color: '{color}', shape: '{shape}', "
            f"size: {node_size}, x: {center_x}, y: {center_y}, title: '{hover_title}'}}"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8"/>
    <title>Mapa Interactivo de Comunidades BotFarm</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #1a1a1a; color: #ffffff; }}
        h2 {{ text-align: center; color: #4dda53; margin-bottom: 10px; }}
        .legend {{ display: flex; justify-content: center; gap: 20px; margin-bottom: 15px; background: #2a2a2a; padding: 10px; border-radius: 8px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 14px; }}
        .dot-box {{ width: 14px; height: 14px; border-radius: 50%; display: inline-block; }}
        #network {{ width: 100%; height: 750px; border: 1px solid #444; border-radius: 8px; background-color: #111111; }}
    </style>
</head>
<body>
    <h2>Islas de Comunidades y Granjas de Bots (Vis.js Grid Layout)</h2>
    <div class="legend">
        <div class="legend-item"><span class="dot-box" style="background:#d9534f;"></span> Q I: Botnet Masiva Crítica</div>
        <div class="legend-item"><span class="dot-box" style="background:#f0ad4e;"></span> Q II: Bot Infiltrado Local</div>
        <div class="legend-item"><span class="dot-box" style="background:#9b59b6;"></span> Q III: Anomalía Global</div>
        <div class="legend-item"><span class="dot-box" style="background:#5cb85c;"></span> Q IV: Humano Orgánico</div>
    </div>
    <div id="network"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet([{", ".join(nodes_js)}]);
        var edges = new vis.DataSet([]);
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                font: {{ color: '#ffffff', size: 12 }},
                borderWidth: 1,
                borderColor: '#ffffff'
            }},
            physics: {{ enabled: false }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html_content, encoding="utf-8")
    return str(out_p)