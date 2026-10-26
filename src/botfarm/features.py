from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .config import Config
from .embedders import Embedder
from .io import to_epoch_seconds
from .pit import PitWindow
from .poisson import burst_stats, creation_burst, iat_anomaly_ratio

# Columnas que alimentan al detector
FEATURES = [
    "density",
    "avg_sim",
    "avg_iat_std",
    "iat_anomaly_ratio",
    "msg_burst_share",
    "msg_fano",
    "creation_std_hours",
    "creation_burst_frac",
    "ttr",
]

META = ["comm_id", "n_members", "n_tweets", "creation_burst_flag", "members"]


def user_profiles(
    tw: pd.DataFrame, embedder: Embedder
) -> Dict[str, np.ndarray]:
  """Embedding medio L2-normalizado por usuario en la ventana actual."""
  if tw.empty:
    return {}

  vecs = np.asarray(
      embedder.encode(tw["text"].astype(str).tolist()), dtype=float
  )
  out = {}

  for u, pos in tw.groupby("user_id", sort=False).indices.items():
    v = vecs[pos].mean(axis=0)
    norm = np.linalg.norm(v)
    out[str(u)] = v / norm if norm > 0 else v

  return out


def ttr_score(texts: pd.Series, max_tokens: int) -> float:
  """Calcula el Type-Token Ratio (TTR) truncado a N tokens."""
  tokens: List[str] = []
  for t in texts:
    tokens.extend(str(t).lower().split())
    if len(tokens) >= max_tokens:
      break

  tokens = tokens[:max_tokens]
  return float(len(set(tokens)) / len(tokens)) if tokens else float("nan")


def community_features(
    tw: pd.DataFrame,
    users: pd.DataFrame,
    G: nx.Graph,
    partition: Dict[str, int],
    user_vec: Dict[str, np.ndarray],
    window: PitWindow,
    cfg: Config,
) -> pd.DataFrame:
  cols = META[:4] + FEATURES + ["members"]

  if tw.empty or not partition:
    return pd.DataFrame(columns=cols)

  # Normalización de IDs a string en partition
  norm_partition = {str(k): int(v) for k, v in partition.items()}

  # Filtrar tweets cuyos usuarios pertenecen a la partición
  tw_proc = tw.copy()
  tw_proc["user_id_str"] = tw_proc["user_id"].astype(str)
  tw_proc = tw_proc[tw_proc["user_id_str"].isin(norm_partition)].copy()

  if tw_proc.empty:
    return pd.DataFrame(columns=cols)

  tw_proc["comm"] = tw_proc["user_id_str"].map(norm_partition).astype(int)
  tw_proc["ts"] = to_epoch_seconds(tw_proc["created_at"])

  created = dict(
      zip(users["user_id"].astype(str), users["created_ts"].astype(float))
  )

  nodes_by_comm = defaultdict(list)
  for node, comm in norm_partition.items():
    nodes_by_comm[comm].append(node)

  t0, t1 = window.start.timestamp(), window.as_of.timestamp()
  finite_ts = np.array([v for v in created.values() if np.isfinite(v)])
  span_s = (t1 - finite_ts.min()) if finite_ts.size > 0 else (t1 - t0)

  rows = []
  for c, g in tw_proc.groupby("comm", sort=True):
    members = sorted(str(u) for u in g["user_id_str"].unique())
    n = len(members)

    if n < cfg.min_members:
      continue

    # A) Semántica: promedio de similitud coseno entre pares de usuarios
    vecs_comm = [user_vec[u] for u in members if u in user_vec]
    if len(vecs_comm) > 1:
      V = np.vstack(vecs_comm)
      s = V.sum(axis=0)
      avg_sim = float((s @ s - len(vecs_comm)) / (len(vecs_comm) * (len(vecs_comm) - 1)))
      avg_sim = float(np.clip(avg_sim, -1.0, 1.0))
    else:
      avg_sim = 0.0

    # B) Grafo: densidad del subgrafo de la comunidad
    comm_nodes = [str(node) for node in nodes_by_comm[c]]
    sub_g = G.subgraph(comm_nodes)
    density = float(nx.density(sub_g)) if sub_g.number_of_nodes() > 1 else 0.0

    # C) Telemetría temporal e intervalos inter-arribo (IAT)
    stds, ratios = [], []
    for _, gu in g.groupby("user_id_str"):
      ts = np.sort(gu["ts"].to_numpy())
      if len(ts) >= cfg.min_tweets_per_user:
        stds.append(np.diff(ts).std())
        ratios.append(iat_anomaly_ratio(ts, cfg.burst_sigma))

    burst_share, fano = burst_stats(
        g["ts"].to_numpy(),
        t0,
        t1,
        cfg.msg_bin_minutes * 60,
        cfg.burst_sigma,
        cfg.min_burst_count,
    )

    # D) Sincronía en fechas de creación de cuentas
    cts = np.array([created.get(u, np.nan) for u in members], dtype=float)
    cts = cts[np.isfinite(cts)]
    creation_std_hours = float(cts.std() / 3600.0) if cts.size > 1 else np.nan

    cb_frac, cb_flag = creation_burst(
        cts,
        cfg.creation_bin_hours * 3600,
        span_s,
        cfg.burst_sigma,
        cfg.min_creation_burst,
    )

    rows.append({
        "comm_id": int(c),
        "n_members": n,
        "n_tweets": len(g),
        "density": density,
        "avg_sim": avg_sim,
        "avg_iat_std": float(np.mean(stds)) if stds else np.nan,
        "iat_anomaly_ratio": float(np.mean(ratios)) if ratios else np.nan,
        "msg_burst_share": burst_share,
        "msg_fano": fano,
        "creation_std_hours": creation_std_hours,
        "creation_burst_frac": cb_frac,
        "creation_burst_flag": cb_flag,
        "ttr": ttr_score(g["text"], cfg.ttr_max_tokens),
        "members": tuple(members),
    })

  return pd.DataFrame(rows, columns=cols)