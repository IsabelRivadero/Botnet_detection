from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
import pandas as pd

from botfarm.config import Config
from botfarm.detector import QuadrantDetector
from botfarm.embedders import HashingEmbedder
from botfarm.io import load_dataset
from botfarm.pipeline import build_features
from .schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    CommunityDetail,
    FitRequest,
    HealthResponse,
    ScoreRequest,
)
from botfarm.visualization import generate_network_html, plot_quadrants, plot_radar_explanation

router = APIRouter()


def resolve_path(p: str | Path) -> Path:
    """Resuelve rutas relativas buscando tanto en el CWD como en la raíz de botfarm."""
    path = Path(p)
    if path.exists():
        return path
    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root / path
    if candidate.exists():
        return candidate
    return path


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/fit")
def fit_model(req: FitRequest):
    try:
        data_path = resolve_path(req.data_dir)
        cfg = Config(lookback_days=req.lookback_days, alpha=req.alpha)
        emb = HashingEmbedder()
        users, tweets, _ = load_dataset(data_path)
        ref_features = build_features(tweets, users, req.as_of, cfg, emb)
        detector = QuadrantDetector(cfg).fit(ref_features)

        model_p = resolve_path(req.model_path)
        if not model_p.parent.exists():
            model_p = Path(req.model_path)

        detector.save(str(model_p))
        return {
            "status": "success",
            "fitted_samples": len(ref_features),
            "model_path": str(model_p),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score")
def score_data(req: ScoreRequest):
    try:
        model_p = resolve_path(req.model_path)
        if not model_p.exists():
            raise HTTPException(
                status_code=404, detail=f"Modelo {req.model_path} no encontrado"
            )
        detector = QuadrantDetector.load(str(model_p))
        cfg = detector.config
        emb = HashingEmbedder()

        data_path = resolve_path(req.data_dir)
        users, tweets, _ = load_dataset(data_path)
        features = build_features(tweets, users, req.as_of, cfg, emb)
        scored = detector.score(features)
        result = pd.concat([features, scored], axis=1)

        out_p = resolve_path(req.out_path)
        result.to_csv(out_p, index=True)
        return {
            "status": "success",
            "evaluated_communities": len(result),
            "output_path": str(out_p),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_stream(req: AnalyzeRequest):
    try:
        model_p = resolve_path(req.model_path)
        if not model_p.exists():
            raise HTTPException(
                status_code=404, detail=f"Modelo {req.model_path} no encontrado"
            )

        detector = QuadrantDetector.load(str(model_p))
        cfg = detector.config
        emb = HashingEmbedder()

        users_df = pd.DataFrame([u.model_dump() for u in req.users])
        if not users_df.empty:
            users_df["user_id"] = users_df["user_id"].astype(str)
            users_df["created_at"] = pd.to_datetime(users_df["created_at"], utc=True)
            users_df["created_ts"] = (
                users_df["created_at"] - pd.Timestamp("1970-01-01", tz="UTC")
            ) / pd.Timedelta(seconds=1)

        tweets_data = []
        for t in req.tweets:
            d = t.model_dump()
            d["mentions"] = d.get("mentions") or []
            tweets_data.append(d)
        tweets_df = pd.DataFrame(tweets_data)
        if not tweets_df.empty:
            tweets_df["user_id"] = tweets_df["user_id"].astype(str)
            tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], utc=True)

        if users_df.empty or tweets_df.empty:
            return AnalysisResponse(
                status="success",
                total_communities=0,
                flagged_communities=0,
                communities=[],
            )

        features = build_features(tweets_df, users_df, req.as_of, cfg, emb)
        if features.empty:
            return AnalysisResponse(
                status="success",
                total_communities=0,
                flagged_communities=0,
                communities=[],
            )

        scored = detector.score(features)
        df_res = pd.concat([features, scored], axis=1)

        communities_list: List[CommunityDetail] = []
        flagged_count = 0

        for cid, row in df_res.iterrows():
            quadrant = str(row.get("quadrant", row.get("cuadrante", "IV_NORMAL")))
            is_bot = quadrant in [
                "I_EXTREMA",
                "II_GRUPO_LOCAL",
                "III_LEJANA_GLOBAL",
            ]
            if is_bot:
                flagged_count += 1

            members = row.get("members", [])
            if hasattr(members, "tolist"):
                members = members.tolist()
            elif isinstance(members, str):
                members = members.split()

            c_detail = CommunityDetail(
                community_id=int(cid),
                size=int(row.get("size", len(members))),
                quadrant=quadrant,
                is_botnet=is_bot,
                scores={
                    "if_score": float(row.get("score_if", 0.0)),
                    "lof_score": float(row.get("score_lof", 0.0)),
                },
                members=[int(m) for m in members if str(m).isdigit()],
            )
            communities_list.append(c_detail)

        return AnalysisResponse(
            status="success",
            total_communities=len(communities_list),
            flagged_communities=flagged_count,
            communities=communities_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/visualize")
def generate_reports(input_csv: str = "flagged_communities.csv"):
    try:
        data_p = resolve_path(input_csv)
        if not data_p.exists():
            raise HTTPException(
                status_code=404, detail=f"Archivo de entrada {input_csv} no encontrado"
            )

        scored_df = pd.read_csv(data_p, index_col=0)
        reports_dir = Path(__file__).resolve().parents[3] / "reports"
        reports_dir.mkdir(exist_ok=True)

        quad_path = reports_dir / "cuadrantes_if_lof.png"
        plot_quadrants(scored_df, output_path=quad_path)

        radar_path = reports_dir / "radar_explicabilidad.png"
        plot_radar_explanation(scored_df, output_path=radar_path)

        html_path = reports_dir / "red_botnets_islas_grid.html"
        generate_network_html(scored_df, output_path=html_path)

        return {
            "status": "success",
            "reports": {
                "quadrants_image": "/reports/cuadrantes_if_lof.png",
                "radar_image": "/reports/radar_explicabilidad.png",
                "interactive_network": "/reports/red_botnets_islas_grid.html",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))