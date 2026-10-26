"""Garantías Point-in-Time (PIT).

Toda lectura y filtrado de datos pasa por PitWindow.
Regla fundamental: Ningún dato con timestamp > as_of puede influir en las características ni en el entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def to_utc(ts: Any) -> pd.Timestamp:
  """Convierte de forma segura cualquier timestamp, fecha o cadena a pd.Timestamp en UTC."""
  timestamp = pd.Timestamp(ts)
  if timestamp.tzinfo is None:
    return timestamp.tz_localize("UTC")
  return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class PitWindow:
  as_of: pd.Timestamp
  start: pd.Timestamp

  @classmethod
  def make(cls, as_of: Any, lookback_days: float) -> "PitWindow":
    as_of_utc = to_utc(as_of)
    start_utc = as_of_utc - pd.Timedelta(days=lookback_days)
    return cls(as_of=as_of_utc, start=start_utc)

  def slice_tweets(self, tweets: pd.DataFrame) -> pd.DataFrame:
    """Filtra los tweets en el intervalo semiabierto (start, as_of]."""
    if tweets.empty or "created_at" not in tweets.columns:
      return pd.DataFrame(columns=tweets.columns)

    mask = (tweets["created_at"] > self.start) & (
        tweets["created_at"] <= self.as_of
    )
    return tweets[mask]

  def slice_users(self, users: pd.DataFrame) -> pd.DataFrame:
    """Filtra los usuarios que ya existían en as_of. Fecha desconocida (NaT) se conserva."""
    if users.empty or "created_at" not in users.columns:
      return users

    mask = users["created_at"].isna() | (users["created_at"] <= self.as_of)
    return users[mask]


def assert_no_future(df: pd.DataFrame, col: str, as_of: pd.Timestamp) -> None:
  """Verifica en tiempo de ejecución que no existan timestamps posteriores a as_of."""
  if df.empty or col not in df.columns:
    return

  max_ts = df[col].max()
  if pd.notna(max_ts) and max_ts > as_of:
    raise AssertionError(
        f"Temporal Data Leak Detectado: la columna '{col}' tiene un valor máximo"
        f" {max_ts} > as_of {as_of}"
    )