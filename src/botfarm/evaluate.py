from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def attach_truth(scored: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
  """Asocia etiquetas reales a cada comunidad calculando la cobertura y la proporción de bots."""
  if scored.empty:
    return scored.assign(
        label_coverage=pd.Series(dtype=float), truth_share=pd.Series(dtype=float)
    )

  lab = {str(k): int(v) for k, v in labels.items()}
  shares, cov = [], []

  for members in scored["members"]:
    members_str = [str(u) for u in members]
    known = [lab[u] for u in members_str if u in lab]

    if members_str:
      cov.append(len(known) / len(members_str))
      shares.append(float(np.mean(known)) if known else np.nan)
    else:
      cov.append(0.0)
      shares.append(np.nan)

  return scored.assign(label_coverage=cov, truth_share=shares)


def evaluate(
    scored: pd.DataFrame,
    labels: pd.Series,
    min_coverage: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
  """Evalúa el desempeño del detector a nivel comunidad."""
  df = attach_truth(scored, labels)
  kept = df[df["label_coverage"] >= min_coverage].copy()

  if kept.empty:
    if verbose:
      print("AVISO: No hay comunidades que cumplan el criterio de cobertura.")
    return {"data": kept, "cuadrantes": pd.DataFrame()}

  kept["y_true"] = (kept["truth_share"] >= 0.5).astype(int)

  if verbose:
    print(
        f"Comunidades totales: {len(df)} | Con cobertura >="
        f" {min_coverage:.0%}: {len(kept)} | Cobertura media:"
        f" {df['label_coverage'].mean():.1%}"
    )
    print(
        f"Etiquetas reales -> Humanas: {(kept.y_true == 0).sum()} | Bots:"
        f" {(kept.y_true == 1).sum()}"
    )

  if kept["y_true"].nunique() < 2 and verbose:
    print(
        "AVISO: La evaluación no contiene ambas clases. Verifique el rango de"
        " fechas o los IDs."
    )

  rules = {
      "union (IF o LOF)": kept["flag_if"] | kept["flag_lof"],
      "interseccion (IF y LOF)": kept["flag_if"] & kept["flag_lof"],
      "solo IF": kept["flag_if"],
      "solo LOF": kept["flag_lof"],
  }

  out: Dict[str, Any] = {}
  for name, pred in rules.items():
    rep = classification_report(
        kept["y_true"],
        pred.astype(int),
        labels=[0, 1],
        target_names=["HUMANO", "BOT"],
        zero_division=0,
        output_dict=True,
    )
    out[name] = {
        k: round(rep["BOT"][k], 3) for k in ("precision", "recall", "f1-score")
    }

    if verbose:
      print(f"\n=== Regla: {name} ===")
      print(
          classification_report(
              kept["y_true"],
              pred.astype(int),
              labels=[0, 1],
              target_names=["HUMANO", "BOT"],
              zero_division=0,
          )
      )
      print(confusion_matrix(kept["y_true"], pred.astype(int), labels=[0, 1]))

  out["cuadrantes"] = pd.crosstab(kept["quadrant"], kept["y_true"])
  if verbose:
    print("\nDistribución de cuadrantes vs Verdad Terrenal:\n", out["cuadrantes"])

  out["data"] = kept
  return out


def evaluate_users(
    scored: pd.DataFrame,
    labels: pd.Series,
    active_users: Iterable[Any],
    verbose: bool = True,
) -> Dict[str, Any]:
  """Evalúa el desempeño a nivel cuenta individual."""
  active_str = [str(u) for u in active_users]
  labels_norm = labels.copy()
  labels_norm.index = labels_norm.index.astype(str)

  lab = labels_norm.reindex(pd.Index(active_str)).dropna().astype(int)

  if lab.empty:
    if verbose:
      print("AVISO: No se encontraron usuarios etiquetados activos en la ventana.")
    return {"bots_en_comunidades": 0.0}

  scored_users = {str(u) for members in scored["members"] for u in members}

  n_bots = int((lab == 1).sum())
  n_hum = int((lab == 0).sum())
  bots_in = int(lab[lab == 1].index.isin(list(scored_users)).sum())

  if verbose:
    print(
        f"\nCuentas etiquetadas activas en la ventana -> Humanas: {n_hum:,} |"
        f" Bots: {n_bots:,}"
    )
    print(
        "Cuentas bot en comunidades puntuadas:"
        f" {bots_in:,} ({bots_in / max(n_bots, 1):.1%})"
    )

  rules = {
      "union (IF o LOF)": scored["flag_if"] | scored["flag_lof"],
      "interseccion (IF y LOF)": scored["flag_if"] & scored["flag_lof"],
      "solo IF": scored["flag_if"],
      "solo LOF": scored["flag_lof"],
  }

  out: Dict[str, Any] = {}
  for name, flag in rules.items():
    flagged = {
        str(u)
        for members, f in zip(scored["members"], flag)
        if f
        for u in members
    }
    y_pred = lab.index.isin(list(flagged)).astype(int)

    rep = classification_report(
        lab.values,
        y_pred,
        labels=[0, 1],
        target_names=["HUMANO", "BOT"],
        zero_division=0,
        output_dict=True,
    )
    out[name] = {
        k: round(rep["BOT"][k], 3) for k in ("precision", "recall", "f1-score")
    }

    if verbose:
      print(f"\n=== Evaluación por Cuenta · Regla: {name} ===")
      print(
          classification_report(
              lab.values,
              y_pred,
              labels=[0, 1],
              target_names=["HUMANO", "BOT"],
              zero_division=0,
          )
      )
      print(confusion_matrix(lab.values, y_pred, labels=[0, 1]))

  out["bots_en_comunidades"] = float(bots_in / max(n_bots, 1))
  return out