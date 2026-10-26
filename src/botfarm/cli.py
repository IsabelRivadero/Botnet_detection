import argparse
from pathlib import Path
import pandas as pd
import uvicorn

from botfarm.config import Config
from botfarm.detector import QuadrantDetector
from botfarm.embedders import HashingEmbedder
from botfarm.io import load_dataset
from botfarm.pipeline import build_features
from botfarm.visualization import generate_network_html, plot_quadrants, plot_radar_explanation


def main():
    parser = argparse.ArgumentParser(description="CLI BotFarm - Sirna Sakatta'iinsa Botnet fi Analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ajaja fit
    fit_p = subparsers.add_parser("fit", help="Leenjisa moodelaa daataa wabii irratti")
    fit_p.add_argument("--data-dir", default="data", help="Kallattii daataa")
    fit_p.add_argument("--as-of", required=True, help="Guyyaa yeroo (Point-in-time, ISO format)")
    fit_p.add_argument("--lookback-days", type=int, default=30, help="Guyyoota duubatti ilaalaman")
    fit_p.add_argument("--alpha", type=float, default=0.05, help="Sadarkaa saaxilamummaa (Alpha)")
    fit_p.add_argument("--out-model", default="model.joblib", help="Bakka moodelli itti olka'amu")

    # Ajaja score
    score_p = subparsers.add_parser("score", help="Komuniitiilee madaaluu fi ramaduu")
    score_p.add_argument("--data-dir", default="data", help="Kallattii daataa")
    score_p.add_argument("--as-of", required=True, help="Guyyaa madaallii")
    score_p.add_argument("--lookback-days", type=int, default=30, help="Guyyoota duubatti ilaalaman")
    score_p.add_argument("--model", default="model.joblib", help="Moodela leenji'e")
    score_p.add_argument("--out-csv", default="flagged_communities.csv", help="Faayila bu'aan itti olka'amu")

    # Ajaja visualize
    vis_p = subparsers.add_parser("visualize", help="Gabaasa fakkiilee fi HTML uumuu")
    vis_p.add_argument("--input-csv", default="flagged_communities.csv", help="CSV ramadame")
    vis_p.add_argument("--out-dir", default="reports", help="Faayila bu'aa gabaasaa")

    # Ajaja api
    api_p = subparsers.add_parser("api", help="Tajaajila Uvicorn REST API jalqabsiisuu")
    api_p.add_argument("--host", default="127.0.0.1", help="Teessoo host")
    api_p.add_argument("--port", type=int, default=8000, help="Lakkoofsa port")

    args = parser.parse_args()

    if args.command == "fit":
        cfg = Config(lookback_days=args.lookback_days, alpha=args.alpha)
        emb = HashingEmbedder()
        users, tweets, _ = load_dataset(args.data_dir)
        ref_features = build_features(tweets, users, pd.Timestamp(args.as_of, tz="UTC"), cfg, emb)
        detector = QuadrantDetector(cfg).fit(ref_features)
        detector.save(args.out_model)
        print(f"✅ Moodelli milkaa'inaan olka'ameera: {args.out_model}")

    elif args.command == "score":
        detector = QuadrantDetector.load(args.model)
        cfg = detector.config
        emb = HashingEmbedder()
        users, tweets, _ = load_dataset(args.data_dir)
        features = build_features(tweets, users, pd.Timestamp(args.as_of, tz="UTC"), cfg, emb)
        scored = detector.score(features)
        res = pd.concat([features, scored], axis=1)
        res.to_csv(args.out_csv)
        print(f"✅ Bu'aan madaallii olka'ameera: {args.out_csv}")

    elif args.command == "visualize":
        df = pd.read_csv(args.input_csv, index_col=0)
        out_d = Path(args.out_dir)
        out_d.mkdir(exist_ok=True)
        plot_quadrants(df, output_path=out_d / "cuadrantes_if_lof.png")
        plot_radar_explanation(df, output_path=out_d / "radar_explicabilidad.png")
        generate_network_html(df, output_path=out_d / "red_botnets_islas_grid.html")
        print(f"✅ Gabaasni fakkiilee fi HTML uumameera: {args.out_dir}")

    elif args.command == "api":
        uvicorn.run("botfarm.api.app:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()