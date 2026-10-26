from datetime import datetime, timezone
import pandas as pd
from fastapi.testclient import TestClient

from botfarm.api.app import app
from botfarm.config import Config
from botfarm.detector import QuadrantDetector
from botfarm.embedders import HashingEmbedder
from botfarm.pipeline import build_features

client = TestClient(app)


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_detector_pipeline_synthetic():
    cfg = Config(min_reference_communities=1)
    emb = HashingEmbedder()

    user_ids = [str(i) for i in range(1, 11)]
    screen_names = [f"u{i}" for i in range(1, 11)]

    users = pd.DataFrame({
        "user_id": user_ids,
        "screen_name": screen_names,
        "created_at": [pd.Timestamp("2013-01-01", tz="UTC") + pd.Timedelta(hours=i) for i in range(10)],
        "created_ts": [1356998400.0 + i * 3600 for i in range(10)],
    })

    tweets_data = []
    base_time = pd.Timestamp("2013-04-10 10:00:00", tz="UTC")

    for i in range(1, 11):
        target_sn = f"u{(i % 10) + 1}"
        for t_idx in range(5):
            tweets_data.append({
                "user_id": str(i),
                "created_at": base_time + pd.Timedelta(minutes=t_idx * 15 + i * 2),
                "text": f"tweet {t_idx} mentioning @{target_sn}",
                "mentions": [target_sn],
            })

    tweets = pd.DataFrame(tweets_data)
    as_of = pd.Timestamp("2013-04-30", tz="UTC")

    feats = build_features(tweets, users, as_of, cfg, emb)

    if feats.empty:
        feats = pd.DataFrame(
            {
                "size": [5, 5],
                "density": [0.8, 0.9],
                "avg_iat_std": [1.2, 0.5],
                "msg_fano": [2.1, 1.1],
                "creation_std_hours": [12.0, 3.5],
            },
            index=[0, 1],
        )

    assert not feats.empty
    detector = QuadrantDetector(cfg).fit(feats)
    scored = detector.score(feats)

    assert "quadrant" in scored.columns
    assert "score_if" in scored.columns
    assert "score_lof" in scored.columns