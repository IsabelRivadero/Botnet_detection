"""Generador de datasets sintéticos tipo Cresci.

Simula usuarios humanos orgánicos en clusters interconectados durante todo el período,
y granjas de bots sincronizadas que aparecen en la fase final con textos repetitivos
y creación masiva de cuentas en lote.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from botfarm.io import to_epoch_seconds

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def make_dataset(
    seed: int = 0,
    n_humans: int = 400,
    cluster: int = 10,
    farms: int = 2,
    farm_size: int = 40,
    days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
  rng = np.random.default_rng(seed)
  vocab = np.array([f"w{i}" for i in range(3000)])
  users, tweets, labels = [], [], {}

  # 1. Creación de usuarios humanos
  for i in range(n_humans):
    uid = str(1000 + i)
    created = T0 - pd.Timedelta(
        days=int(rng.integers(30, 1500)), hours=int(rng.integers(0, 24))
    )
    users.append((uid, f"h{i}", created))
    labels[uid] = 0

  # Tweets de usuarios humanos
  for i in range(n_humans):
    uid = str(1000 + i)
    cl = i // cluster
    peers = [j for j in range(cl * cluster, (cl + 1) * cluster) if j != i]
    n_tw = rng.poisson(2 * days)
    secs = np.sort(rng.uniform(0, days * 86400, n_tw))

    for sec in secs:
      txt = " ".join(rng.choice(vocab, size=int(rng.integers(6, 16))))
      if rng.random() < 0.3 and peers:
        txt = f"@h{rng.choice(peers)} " + txt
      tweets.append((uid, txt, T0 + pd.Timedelta(seconds=float(sec))))

  # 2. Creación de granjas de bots
  for f in range(farms):
    templates = [" ".join(rng.choice(vocab, size=8)) for _ in range(8)]
    c0 = T0 + pd.Timedelta(days=17 + f)  # Creación en lote de 6 horas
    ids = []

    for k in range(farm_size):
      uid = str(900000 + f * 1000 + k)
      ids.append(uid)
      users.append((
          uid,
          f"b{f}_{k}",
          c0 + pd.Timedelta(seconds=float(rng.uniform(0, 6 * 3600))),
      ))
      labels[uid] = 1

    # Ráfagas sincronizadas de publicación en los últimos 10 días
    for burst in np.arange(days - 10, days, 1 / 12):
      b0 = T0 + pd.Timedelta(days=float(burst))
      for uid in ids:
        txt = (
            f"RT @cand{f} " + templates[int(rng.integers(0, len(templates)))]
        )
        tweets.append((
            uid,
            txt,
            b0 + pd.Timedelta(seconds=float(rng.uniform(0, 20))),
        ))

  u = pd.DataFrame(users, columns=["user_id", "screen_name", "created_at"])
  u["created_ts"] = to_epoch_seconds(u["created_at"])

  t = pd.DataFrame(tweets, columns=["user_id", "text", "created_at"])
  t = t.sort_values(["created_at", "user_id"], kind="stable").reset_index(
      drop=True
  )

  return u, t, pd.Series(labels, dtype=int)