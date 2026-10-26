"""Construcción de grafos de menciones y detección de comunidades con Louvain."""

from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd

from .config import Config


def build_mention_graph(
    tw: pd.DataFrame, screen_to_id: Dict[str, str]
) -> nx.Graph:
  """Construye un grafo no dirigido de menciones ponderado."""
  G = nx.Graph()

  if tw.empty:
    return G

  # Asegura que todos los IDs de usuario sean tratados como strings uniformes
  active_users = [str(u) for u in tw["user_id"].unique()]
  G.add_nodes_from(active_users)

  m = tw[["user_id", "text"]].copy()
  m["user_id"] = m["user_id"].astype(str)

  m["mention"] = m["text"].astype(str).str.findall(r"@(\w+)")
  m = m.explode("mention").dropna(subset=["mention"])

  if m.empty:
    return G

  m["mention"] = m["mention"].astype(str).str.lower()

  norm_screen_map = {
      str(k).lower(): str(v)
      for k, v in screen_to_id.items()
      if pd.notna(k) and pd.notna(v)
  }

  m["target"] = m["mention"].map(norm_screen_map)
  m["target"] = m["target"].where(m["target"].notna(), "@" + m["mention"])

  # Elimina automenciones
  m = m[m["target"] != m["user_id"]]

  for (u, v), w in m.groupby(["user_id", "target"]).size().items():
    u_str, v_str = str(u), str(v)
    if G.has_edge(u_str, v_str):
      G[u_str][v_str]["weight"] += int(w)
    else:
      G.add_edge(u_str, v_str, weight=int(w))

  return G


def detect_communities(G: nx.Graph, cfg: Config) -> Dict[str, int]:
  """Ejecuta Louvain para encontrar particiones de comunidades en el grafo."""
  if G.number_of_nodes() == 0:
    return {}

  if G.number_of_nodes() == 1:
    node = next(iter(G.nodes()))
    return {str(node): 0}

  comms = nx.community.louvain_communities(
      G,
      weight="weight",
      resolution=cfg.louvain_resolution,
      seed=cfg.seed,
  )

  return {str(node): i for i, comm in enumerate(comms) for node in comm}