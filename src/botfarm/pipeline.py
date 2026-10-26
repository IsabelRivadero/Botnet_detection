"""Orquestador principal de extracción de características Point-in-Time."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Config
from .embedders import Embedder
from .features import community_features, user_profiles
from .graph import build_mention_graph, detect_communities
from .pit import PitWindow, assert_no_future


def build_features(
    tweets: pd.DataFrame,
    users: pd.DataFrame,
    as_of: Any,
    cfg: Config,
    embedder: Embedder,
) -> pd.DataFrame:
  """Extrae características por comunidad en el sello temporal as_of sin leaks temporales."""
  window = PitWindow.make(as_of, cfg.lookback_days)

  tw = (
      window.slice_tweets(tweets)
      .sort_values(["created_at", "user_id"], kind="stable")
      .reset_index(drop=True)
  )
  us = window.slice_users(users)

  assert_no_future(tw, "created_at", window.as_of)
  assert_no_future(us, "created_at", window.as_of)

  if tw.empty:
    return pd.DataFrame()

  screen_to_id = {
      str(s): str(u)
      for s, u in zip(us["screen_name"], us["user_id"])
      if pd.notna(s) and pd.notna(u)
  }

  G = build_mention_graph(tw, screen_to_id)
  partition = detect_communities(G, cfg)

  user_vec = user_profiles(tw, embedder)

  feats = community_features(tw, us, G, partition, user_vec, window, cfg)

  if not feats.empty:
    feats.insert(0, "as_of", window.as_of)

  return feats


def active_users(tweets: pd.DataFrame, as_of: Any, cfg: Config) -> pd.Index:
  """Devuelve el conjunto de usuarios que han publicado al menos un tweet en la ventana actual."""
  window = PitWindow.make(as_of, cfg.lookback_days)
  sliced_tweets = window.slice_tweets(tweets)

  if sliced_tweets.empty:
    return pd.Index([], dtype=str)

  return pd.Index(sliced_tweets["user_id"].astype(str).unique())