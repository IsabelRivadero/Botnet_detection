"""Cálculos de telemetría temporal y ráfagas mediante procesos de Poisson."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def iat_anomaly_ratio(
    ts: np.ndarray, burst_sigma: float = 3.0
) -> float:
  """Calcula la proporción de intervalos inter-arribo (IAT) anómalamente cortos."""
  if len(ts) < 2:
    return float("nan")

  iats = np.diff(np.sort(ts))
  if len(iats) == 0:
    return float("nan")

  mean_iat = np.mean(iats)
  std_iat = np.std(iats)

  if mean_iat <= 0 or std_iat == 0:
    return 1.0

  threshold = max(0.0, mean_iat - burst_sigma * std_iat)
  anomalous_count = np.sum(iats <= threshold)
  return float(anomalous_count / len(iats))


def burst_stats(
    ts: np.ndarray,
    t0: float,
    t1: float,
    bin_seconds: float,
    burst_sigma: float,
    min_burst_count: int,
) -> Tuple[float, float]:
  """Calcula la proporción de mensajes en ráfagas y el Índice de Fano (Var/Mean)."""
  if len(ts) == 0 or t1 <= t0 or bin_seconds <= 0:
    return 0.0, float("nan")

  n_bins = int(np.ceil((t1 - t0) / bin_seconds))
  if n_bins <= 0:
    return 0.0, float("nan")

  counts, _ = np.histogram(ts, bins=n_bins, range=(t0, t1))
  total_msgs = len(ts)
  lam = total_msgs / n_bins if n_bins > 0 else 0.0

  threshold = max(float(min_burst_count), lam + burst_sigma * np.sqrt(lam))

  burst_mask = counts >= threshold
  burst_msgs = counts[burst_mask].sum()
  burst_share = float(burst_msgs / total_msgs) if total_msgs > 0 else 0.0

  mean_c = np.mean(counts)
  var_c = np.var(counts)
  fano = float(var_c / mean_c) if mean_c > 0 else float("nan")

  return burst_share, fano


def creation_burst(
    cts: np.ndarray,
    bin_seconds: float,
    span_seconds: float,
    burst_sigma: float,
    min_creation_burst: int,
) -> Tuple[float, bool]:
  """Detecta ráfagas en las fechas de creación de cuentas de una comunidad."""
  if len(cts) == 0 or span_seconds <= 0 or bin_seconds <= 0:
    return 0.0, False

  n_bins = int(np.ceil(span_seconds / bin_seconds))
  if n_bins <= 0:
    return 0.0, False

  t_min = np.min(cts)
  t_max = t_min + n_bins * bin_seconds

  counts, _ = np.histogram(cts, bins=n_bins, range=(t_min, t_max))
  total_acc = len(cts)
  lam = total_acc / n_bins if n_bins > 0 else 0.0

  threshold = max(
      float(min_creation_burst), lam + burst_sigma * np.sqrt(lam)
  )

  burst_mask = counts >= threshold
  burst_acc = counts[burst_mask].sum()
  cb_frac = float(burst_acc / total_acc) if total_acc > 0 else 0.0
  cb_flag = bool(np.any(burst_mask))

  return cb_frac, cb_flag