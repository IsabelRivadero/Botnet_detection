"""Módulo I/O para carga y normalización sin fugas de etiquetas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterator, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

TW_USER = ("user_id", "user_id_str", "author_id", "user", "author")
TW_TEXT = ("text", "full_text", "tweet", "content")
TW_TIME = ("created_at", "timestamp", "date_time")
US_ID = ("id", "user_id")
US_TIME = ("created_at", "user_created_at")


@dataclass(frozen=True)
class Source:
    path: Path
    label: int  # 0 humano, 1 bot, -1 no etiquetado


def normalize_ids(s: pd.Series) -> pd.Series:
    """Limpia identificadores eliminando sufijos '.0' flotantes y espacios en blanco."""
    return s.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)


_DATE_FORMATS = (
    "%a %b %d %H:%M:%S %z %Y",  # Twitter API v1: Wed Apr 06 17:09:42 +0000 2011
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f%z",
)


def detect_date_format(sample: pd.Series) -> Optional[str]:
    """Detecta el formato strptime si cubre >= 99% de la muestra para parsing optimizado."""
    s = sample.dropna().astype(str).head(2000)
    if s.empty:
        return None

    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
        if parsed.notna().mean() >= 0.99:
            return fmt
    return None


def parse_dates(s: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    """Convierte series a datetime en UTC."""
    if fmt is None:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")

    out = pd.to_datetime(s, errors="coerce", utc=True, format=fmt)
    bad = out.isna() & s.notna()

    if bad.any():
        out = out.copy()
        out[bad] = pd.to_datetime(
            s[bad], errors="coerce", utc=True, format="mixed"
        )
    return out


def to_epoch_seconds(s: pd.Series) -> pd.Series:
    """Convierte timestamps de pandas a segundos epoch numéricos."""
    return (s - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)


def _first(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    return next((c for c in candidates if c in cols), None)


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kw)


def load_users(sources: Sequence[Source]) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga usuarios y separa sus etiquetas ground truth."""
    frames, labels = [], []

    for src in sources:
        f = Path(src.path) / "users.csv"
        if not f.exists():
            warnings.warn(f"No existe el archivo: {f}")
            continue

        df = _read_csv(f, dtype=str, low_memory=False, on_bad_lines="skip")
        idc = _first(df.columns, US_ID)
        tc = _first(df.columns, US_TIME)

        if idc is None:
            warnings.warn(f"Archivo {f} omitido: no contiene columna de identificador.")
            continue

        user_ids = normalize_ids(df[idc])
        out = pd.DataFrame({
            "user_id": user_ids,
            "screen_name": (
                df["screen_name"].str.lower()
                if "screen_name" in df.columns
                else pd.NA
            ),
            "created_at": parse_dates(df[tc]) if tc else pd.NaT,
        }).dropna(subset=["user_id"])

        frames.append(out)
        labels.append(pd.Series(src.label, index=out["user_id"], dtype=int))

    if not frames:
        raise FileNotFoundError("No se pudo cargar ningún archivo users.csv válido.")

    users = pd.concat(frames, ignore_index=True).drop_duplicates("user_id")
    users["created_ts"] = to_epoch_seconds(users["created_at"])

    all_labels = pd.concat(labels)
    lab = all_labels[~all_labels.index.duplicated()]

    return users.reset_index(drop=True), lab


def _iter_tweets(
    path: Path, enc: str, start: pd.Timestamp, end: pd.Timestamp, chunksize: int
) -> Iterator[pd.DataFrame]:
    cols = pd.read_csv(path, nrows=0, encoding=enc).columns
    uc = _first(cols, TW_USER)
    xc = _first(cols, TW_TEXT)
    tc = _first(cols, TW_TIME)

    if not (uc and xc and tc):
        raise ValueError(
            f"Archivo {path} incompleto. Columnas requeridas no halladas:"
            f" {list(cols)}"
        )

    reader = pd.read_csv(
        path,
        usecols=[uc, xc, tc],
        dtype=str,
        chunksize=chunksize,
        encoding=enc,
        on_bad_lines="skip",
    )

    name = path.parent.name
    t_start = time.time()
    fmt = None
    seen, kept = 0, 0

    for i, ch in enumerate(reader):
        ch = ch.rename(columns={uc: "user_id", xc: "text", tc: "created_at"})

        if i == 0:
            fmt = detect_date_format(ch["created_at"])
            print(
                f"  [{name}] Formato de fecha inferido: {fmt or 'mixto (retroceso lento)'}",
                flush=True,
            )

        ch["created_at"] = parse_dates(ch["created_at"], fmt)
        seen += len(ch)

        ch = ch.dropna()
        ch = ch[(ch["created_at"] > start) & (ch["created_at"] <= end)]
        kept += len(ch)

        print(
            f"  [{name}] {seen:>11,} leídas | {kept:>9,} en ventana |"
            f" {time.time() - t_start:5.0f}s",
            flush=True,
        )

        if len(ch) > 0:
            ch["user_id"] = normalize_ids(ch["user_id"])
            yield ch


def load_tweets(
    sources: Sequence[Source],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Carga y filtra tweets en la ventana (start, end]."""
    from .pit import to_utc

    start_utc, end_utc = to_utc(start), to_utc(end)
    parts = []

    for src in sources:
        f = Path(src.path) / "tweets.csv"
        if not f.exists():
            warnings.warn(f"No existe el archivo de tweets: {f}")
            continue

        print(f"Procesando {f}...", flush=True)
        try:
            file_parts = list(_iter_tweets(f, "utf-8", start_utc, end_utc, chunksize))
        except UnicodeDecodeError:
            file_parts = list(
                _iter_tweets(f, "latin1", start_utc, end_utc, chunksize)
            )

        parts.extend(file_parts)

    if not parts:
        raise FileNotFoundError(
            "No se encontraron tweets dentro de la ventana de tiempo especificada."
        )

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["created_at", "user_id"], kind="stable")
        .reset_index(drop=True)
    )


def sample_timestamps(
    sources: Sequence[Source], chunksize: int = 500_000, step: int = 20
) -> Dict[str, Tuple[int, pd.Series]]:
    """Muestra rápida de marcas temporales para diagnósticos y auditar rangos de datos."""
    out = {}

    for src in sources:
        f = Path(src.path) / "tweets.csv"
        if not f.exists():
            continue

        cols = pd.read_csv(f, nrows=0, encoding="latin1").columns
        tc = _first(cols, TW_TIME)

        if tc is None:
            continue

        sample = []
        for ch in pd.read_csv(
            f,
            usecols=[tc],
            dtype=str,
            chunksize=chunksize,
            encoding="latin1",
            on_bad_lines="skip",
        ):
            sample.append(ch[tc].iloc[::step])

        if sample:
            combined = parse_dates(pd.concat(sample)).dropna()
            out[Path(src.path).name] = (src.label, combined)

    return out


def summarize_ranges(
    samples: Dict[str, Tuple[int, pd.Series]]
) -> pd.DataFrame:
    rows = []
    for name, (lab, ts) in samples.items():
        if len(ts) == 0:
            continue
        rows.append({
            "source": name,
            "label": lab,
            "n_sample": len(ts),
            "min": ts.min(),
            "p05": ts.quantile(0.05),
            "median": ts.quantile(0.5),
            "p95": ts.quantile(0.95),
            "max": ts.max(),
        })
    return pd.DataFrame(rows)


def summarize_monthly(
    samples: Dict[str, Tuple[int, pd.Series]], step: int = 20
) -> pd.DataFrame:
    cols = {}
    for name, (_, ts) in samples.items():
        if len(ts) == 0:
            continue
        counts = (
            ts.dt.tz_localize(None).dt.to_period("M").value_counts().sort_index()
            * step
        )
        cols[name] = counts

    return pd.DataFrame(cols).fillna(0).astype(int)


def load_dataset(
    data_dir: str | Path = "data",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carga usuarios, tweets y etiquetas creando objetos Source desde data_dir."""
    base_path = Path(data_dir)
    default_sources = [
        ("genuine_accounts.csv", 0),
        ("social_spambots_1.csv", 1),
        ("social_spambots_2.csv", 1),
        ("social_spambots_3.csv", 1),
        ("traditional_spambots_1.csv", 1),
    ]

    sources = []
    for folder, label in default_sources:
        p = base_path / folder
        if p.exists():
            sources.append(Source(path=p, label=label))

    if not sources and base_path.exists():
        for sub in base_path.iterdir():
            if sub.is_dir() and ((sub / "users.csv").exists() or (sub / "tweets.csv").exists()):
                sources.append(Source(path=sub, label=-1))

    if not sources:
        if (base_path / "users.csv").exists() or (base_path / "tweets.csv").exists():
            sources.append(Source(path=base_path, label=-1))

    if not sources:
        raise FileNotFoundError(f"No se encontraron datos válidos en: {data_dir}")

    users, labels = load_users(sources)

    if start is None:
        start = pd.Timestamp("1970-01-01", tz="UTC")
    if end is None:
        end = pd.Timestamp.now(tz="UTC")

    try:
        tweets = load_tweets(sources, start=start, end=end)
    except FileNotFoundError:
        tweets = pd.DataFrame(columns=["user_id", "text", "created_at"])

    return users, tweets, labels