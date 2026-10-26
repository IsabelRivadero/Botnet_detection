from pathlib import Path
import pandas as pd

from botfarm.config import Config
from botfarm.detector import QuadrantDetector
from botfarm.embedders import HashingEmbedder
from botfarm.io import load_dataset
from botfarm.pipeline import build_features
from botfarm.visualization import (
    generate_network_html,
    plot_quadrants,
    plot_radar_explanation,
)


def main():
    print("🎨 Generando visualizaciones para BotFarm...")

    base_dir = Path(__file__).resolve().parents[1]
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    flagged_csv = base_dir / "flagged_communities.csv"
    model_path = base_dir / "model.joblib"

    if flagged_csv.exists():
        print(f"📄 Cargando datos desde {flagged_csv.name}...")
        scored_df = pd.read_csv(flagged_csv, index_col=0)
    elif model_path.exists():
        print("⚡ Evaluando datos usando model.joblib...")
        detector = QuadrantDetector.load(model_path)
        cfg = detector.config
        emb = HashingEmbedder()
        users, tweets, _ = load_dataset(base_dir / "data")
        as_of = pd.Timestamp("2013-09-30", tz="UTC")
        features = build_features(tweets, users, as_of, cfg, emb)
        scored = detector.score(features)
        scored_df = pd.concat([features, scored], axis=1)
    else:
        print("⚠️ No se encontró flagged_communities.csv ni model.joblib.")
        return

    # 1. Matriz de Cuadrantes 2D
    quad_path = reports_dir / "cuadrantes_if_lof.png"
    plot_quadrants(scored_df, output_path=quad_path, thr_if=0.8, thr_lof=0.8)
    print(f"✅ Matriz de Cuadrantes guardada en: {quad_path}")

    # 2. Radar Chart de Explicabilidad
    radar_path = reports_dir / "radar_explicabilidad.png"
    plot_radar_explanation(scored_df, output_path=radar_path)
    print(f"✅ Radar Chart guardado en: {radar_path}")

    # 3. Grafo Interactivo HTML
    html_path = reports_dir / "red_botnets_islas_grid.html"
    generate_network_html(scored_df, output_path=html_path)
    print(f"✅ Mapa Interactivo HTML generado en: {html_path}")

    print("\n🎉 Proceso de visualización finalizado.")


if __name__ == "__main__":
    main()