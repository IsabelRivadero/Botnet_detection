"""Detector por cuadrantes IF x LOF con estructura fit(pasado) / score(futuro).

- IsolationForest  -> anomalías LEJANAS (aislamiento global)
- LOF (novelty)    -> anomalías de GRUPO (densidad local baja respecto a sus vecinos)
- Intersección     -> anomalías EXTREMAS
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import FEATURES

LOG_COLS = ["avg_iat_std", "msg_fano", "creation_std_hours"]
Q_EXTREME = "I_EXTREMA"
Q_LOCAL = "II_GRUPO_LOCAL"
Q_GLOBAL = "III_LEJANA_GLOBAL"
Q_NORMAL = "IV_NORMAL"


class QuadrantDetector:

  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.imputer_: SimpleImputer | None = None
    self.scaler_: StandardScaler | None = None
    self.iforest_: IsolationForest | None = None
    self.lof_: LocalOutlierFactor | None = None
    self.thr_if_: float = 0.0
    self.thr_lof_: float = 0.0
    self.ref_if_: np.ndarray = np.array([])
    self.ref_lof_: np.ndarray = np.array([])
    self.ref_as_of_: pd.Timestamp | None = None

  @property
  def config(self) -> Config:
    return self.cfg

  def _prep(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
    if X.empty:
      return np.empty((0, len(FEATURES)))

    cols_present = [c for c in FEATURES if c in X.columns]
    X_proc = X[cols_present].astype(float).copy()

    for col in FEATURES:
      if col not in X_proc.columns:
        X_proc[col] = np.nan

    X_proc = X_proc[FEATURES]

    for col in LOG_COLS:
      if col in X_proc.columns:
        X_proc[col] = np.log1p(X_proc[col].clip(lower=0))

    if fit:
      self.imputer_ = SimpleImputer(
          strategy="median", keep_empty_features=True
      ).fit(X_proc)
      X_imp = self.imputer_.transform(X_proc)
      self.scaler_ = StandardScaler().fit(X_imp)
      return self.scaler_.transform(X_imp)

    if self.imputer_ is None or self.scaler_ is None:
      raise RuntimeError(
          "El detector no ha sido entrenado. Llame a .fit() antes de .score()."
      )

    X_imp = self.imputer_.transform(X_proc)
    return self.scaler_.transform(X_imp)

  def fit(self, ref: pd.DataFrame) -> "QuadrantDetector":
    if len(ref) < self.cfg.min_reference_communities:
      raise ValueError(
          f"Referencia con {len(ref)} comunidades; mínimo requerido"
          f" {self.cfg.min_reference_communities}."
      )

    Xs = self._prep(ref, fit=True)
    self.iforest_ = IsolationForest(
        n_estimators=self.cfg.if_estimators, random_state=self.cfg.seed
    ).fit(Xs)

    k = max(1, min(self.cfg.lof_neighbors, max(1, len(Xs) - 1)))
    self.lof_ = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xs)

    ref_if = -self.iforest_.score_samples(Xs)
    ref_lof = -self.lof_.negative_outlier_factor_

    self.ref_if_ = np.sort(ref_if)
    self.ref_lof_ = np.sort(ref_lof)

    q = 1.0 - self.cfg.alpha
    self.thr_if_ = float(np.quantile(ref_if, q))
    self.thr_lof_ = float(np.quantile(ref_lof, q))

    return self

  @staticmethod
  def _pct(sorted_ref: np.ndarray, s: np.ndarray) -> np.ndarray:
    if len(sorted_ref) == 0 or len(s) == 0:
      return np.array([])
    return np.searchsorted(sorted_ref, s, side="right") / len(sorted_ref)

  def score(self, X: pd.DataFrame) -> pd.DataFrame:
    if self.iforest_ is None or self.lof_ is None:
      raise RuntimeError(
          "El modelo no ha sido ajustado. Llame a .fit() antes de .score()."
      )

    if X.empty:
      return pd.DataFrame(
          columns=[
              "score_if",
              "score_lof",
              "pct_if",
              "pct_lof",
              "flag_if",
              "flag_lof",
              "quadrant",
              "risk",
          ],
          index=X.index,
      )

    Xs = self._prep(X)
    s_if = -self.iforest_.score_samples(Xs)
    s_lof = -self.lof_.score_samples(Xs)

    f_if = s_if >= self.thr_if_
    f_lof = s_lof >= self.thr_lof_

    quad = np.select(
        [f_if & f_lof, f_lof, f_if],
        [Q_EXTREME, Q_LOCAL, Q_GLOBAL],
        default=Q_NORMAL,
    )

    pct_if = self._pct(self.ref_if_, s_if)
    pct_lof = self._pct(self.ref_lof_, s_lof)

    return pd.DataFrame(
        {
            "score_if": s_if,
            "score_lof": s_lof,
            "pct_if": pct_if,
            "pct_lof": pct_lof,
            "flag_if": f_if,
            "flag_lof": f_lof,
            "quadrant": quad,
            "risk": 100.0 * (pct_if + pct_lof) / 2.0,
        },
        index=X.index,
    )

  def save(self, path: Union[str, Path]) -> None:
    joblib.dump(self, path)

  @staticmethod
  def load(path: Union[str, Path]) -> "QuadrantDetector":
    return joblib.load(path)