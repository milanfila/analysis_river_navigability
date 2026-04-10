from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

META_URL = "https://opendata.chmi.cz/hydrology/historical/metadata/meta1.json"
LOCAL_META_PATH = Path("hydro_meta1.json")


def _parse_meta_json(obj: dict) -> pd.DataFrame:
    """
    Převede CHMI metadata JSON -> dataframe.
    Očekává strukturu:
      obj["data"]["data"]["header"]
      obj["data"]["data"]["values"]
    """
    data_block = obj.get("data", {}).get("data", {})
    header = data_block.get("header")
    values = data_block.get("values", [])

    if not header or not values:
        raise ValueError("Metadata JSON neobsahuje očekávané header/values.")

    cols = [c.strip() for c in header.split(",")]
    df = pd.DataFrame(values, columns=cols)

    # přejmenujeme si nejdůležitější sloupce do praktičtějších názvů
    rename_map = {
        "objID": "station_id",
        "DBC": "station_code",
        "STATION_NAME": "station_name",
        "STREAM_NAME": "stream_name",
        "GEOGR1": "lat",
        "GEOGR2": "lon",
        "SPA_TYP": "spa_type",
        "SPAH_DS": "stage_desc",
        "SPAH_UNIT": "stage_unit",
        "DRYH": "dry_h",
        "SPA1H": "spa1_h",
        "SPA2H": "spa2_h",
        "SPA3H": "spa3_h",
        "SPA4H": "spa4_h",
        "SPAQ_DS": "flow_desc",
        "SPAQ_UNIT": "flow_unit",
        "DRYQ": "dry_q",
        "SPA1Q": "spa1_q",
        "SPA2Q": "spa2_q",
        "SPA3Q": "spa3_q",
        "SPA4Q": "spa4_q",
        "PLO_STA": "catchment_area_km2",
        "HLGP4": "basin_code",
    }
    df = df.rename(columns=rename_map)

    # numerické sloupce
    num_cols = [
        "lat", "lon",
        "dry_h", "spa1_h", "spa2_h", "spa3_h", "spa4_h",
        "dry_q", "spa1_q", "spa2_q", "spa3_q", "spa4_q",
        "catchment_area_km2",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # standardizace textů
    for c in ["station_id", "station_code", "station_name", "stream_name", "spa_type", "stage_unit", "flow_unit", "basin_code"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # pomocný label pro UI
    df["label"] = (
        df["stream_name"].fillna("") + " | " +
        df["station_name"].fillna("") + " | " +
        df["station_id"].fillna("")
    )

    return df


def load_station_catalog(
    meta_url: str = META_URL,
    local_meta_path: Optional[Path] = LOCAL_META_PATH,
    timeout: int = 30,
) -> tuple[pd.DataFrame, str]:
    """
    Načte katalog stanic:
    1) primárně z webu
    2) fallback z lokálního souboru

    Vrací:
      df, source
    kde source je "web" nebo "local".
    """
    # 1) web
    try:
        r = requests.get(meta_url, timeout=timeout)
        r.raise_for_status()
        obj = r.json()
        df = _parse_meta_json(obj)
        return df, "web"
    except Exception:
        pass

    # 2) local fallback
    if local_meta_path is None or not Path(local_meta_path).exists():
        raise FileNotFoundError(
            f"Nepodařilo se načíst metadata z webu a lokální fallback neexistuje: {local_meta_path}"
        )

    with open(local_meta_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    df = _parse_meta_json(obj)
    return df, "local"


def prepare_station_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Připraví dataframe pro UI:
    - jen stanice s vyplněným station_id, stream_name, station_name
    - seřazení podle řeky a názvu
    """
    out = df.copy()

    out = out.dropna(subset=["station_id", "stream_name", "station_name"]).copy()
    out = out[out["station_id"].astype(str).str.len() > 0].copy()
    out = out[out["stream_name"].astype(str).str.len() > 0].copy()
    out = out[out["station_name"].astype(str).str.len() > 0].copy()

    out = out.sort_values(["stream_name", "station_name", "station_id"]).reset_index(drop=True)
    return out


def filter_by_stream(df: pd.DataFrame, stream_name: str) -> pd.DataFrame:
    return df[df["stream_name"] == stream_name].copy()