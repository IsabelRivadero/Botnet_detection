from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Config:
  # --- Ventana point-in-time ---
  lookback_days: float = (
      7.0  # Ancho de la ventana de observación (as_of - lookback, as_of]
  )

  # --- Comunidades ---
  min_members: int = (
      3  # Mínimo de miembros ACTIVOS (con tweets en la ventana)
  )
  louvain_resolution: float = 1.0

  # --- Temporalidad / Poisson ---
  min_tweets_per_user: int = 3  # Mínimo de tweets para calcular IAT por usuario
  msg_bin_minutes: int = 10  # Ancho del bin para ráfagas de mensajes
  creation_bin_hours: int = (
      24  # Ancho del bin para ráfagas de creación de cuentas
  )
  burst_sigma: float = 3.0  # Umbral Poisson: lambda + sigma * sqrt(lambda)
  min_burst_count: int = (
      5  # Piso: evita falsos positivos con tasas lambda cercanas a 0
  )
  min_creation_burst: int = 3

  # --- Léxico ---
  ttr_max_tokens: int = 2000  # TTR sobre N tokens fijos

  # --- Detector ---
  if_estimators: int = 300
  lof_neighbors: int = 20
  alpha: float = (
      0.05  # Tasa de falsos positivos objetivo sobre la ventana de referencia
  )
  min_reference_communities: int = 10

  seed: int = 42

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Config":
    valid_keys = cls.__dataclass_fields__.keys()
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)