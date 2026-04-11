from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Hydro multiprofile dashboard", page_icon="🌊", layout="wide")

LOCAL_TZ = "Europe/Prague"

BASE_NOW = "https://opendata.chmi.cz/hydrology/now/data"
BASE_HIST = "https://opendata.chmi.cz/hydrology/historical/data/hourly"
META_URL = "https://opendata.chmi.cz/hydrology/historical/metadata/meta1.json"

LOCAL_META_PATHS = [
    Path("hydro_meta1.json"),
    Path("../hydro_meta1.json"),
]

DEFAULT_MODEL_PATHS = [
    Path("oslava_model_rise_tplus2h.json"),
    Path("models/oslava_model_rise_tplus2h.json"),
    Path("basic_dashboard/oslava_model_rise_tplus2h.json"),
]

MODELS_DIR = Path("models")
LOCAL_SAVE_ENABLED = True

RECOMMENDED_PAIRS = [
    {
        "pair_id": "oslava_mostiste_nesmer",
        "label": "Oslava | VD Mostiště → Nesměř",
        "stream_name": "Oslava",
        "upstream_id": "0-203-1-471000",
        "downstream_id": "0-203-1-473000",
        "description": "Krátký regulovaný úsek pod VD Mostiště vhodný pro kajak model.",
        "model_key": "oslava_rise_tplus2h",
    },
    {
        "pair_id": "oslava_nesmer_oslavany",
        "label": "Oslava | Nesměř → Oslavany",
        "stream_name": "Oslava",
        "upstream_id": "0-203-1-473000",
        "downstream_id": "0-203-1-474000",
        "description": "Delší downstream úsek Oslavy, vhodný pro experimentální lag analýzu.",
        "model_key": None,
    },
    {
        "pair_id": "jihlava_ptacov_mohelno",
        "label": "Jihlava | Třebíč-Ptáčov → VD Mohelno",
        "stream_name": "Jihlava",
        "upstream_id": "0-203-1-469000",
        "downstream_id": "0-203-1-469500",
        "description": "Příklad doporučené dvojice na Jihlavě.",
        "model_key": None,
    },
    {
        "pair_id": "svratka_vir_bityska",
        "label": "Svratka | VD Vír pod vyrovnávací nádrží → Veverská Bítýška",
        "stream_name": "Svratka",
        "upstream_id": "0-203-1-445000",
        "downstream_id": "0-203-1-448000",
        "description": "Regulovaný tok pod vodním dílem Vír.",
        "model_key": None,
    },
]


# =========================================================
# OBECNÉ POMOCNÉ FUNKCE
# =========================================================
def parse_dt_to_local(series, local_tz: str = LOCAL_TZ) -> pd.DatetimeIndex:
    dt = pd.to_datetime(pd.Series(series).astype(str), utc=True, errors="coerce")
    return pd.DatetimeIndex(dt).tz_convert(local_tz)


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def get_session_model_key(pair_id: Optional[str]) -> Optional[str]:
    if not pair_id:
        return None
    return f"active_model::{pair_id}"


# =========================================================
# METADATA STANIC
# =========================================================
def _parse_meta_json(obj: dict) -> pd.DataFrame:
    data_block = obj.get("data", {}).get("data", {})
    header = data_block.get("header")
    values = data_block.get("values", [])

    if not header or not values:
        raise ValueError("Metadata JSON neobsahuje očekávané header/values.")

    cols = [c.strip() for c in header.split(",")]
    df = pd.DataFrame(values, columns=cols)

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

    num_cols = [
        "lat",
        "lon",
        "dry_h",
        "spa1_h",
        "spa2_h",
        "spa3_h",
        "spa4_h",
        "dry_q",
        "spa1_q",
        "spa2_q",
        "spa3_q",
        "spa4_q",
        "catchment_area_km2",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    text_cols = [
        "station_id",
        "station_code",
        "station_name",
        "stream_name",
        "spa_type",
        "stage_desc",
        "stage_unit",
        "flow_desc",
        "flow_unit",
        "basin_code",
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["label"] = (
        df["stream_name"].fillna("")
        + " | "
        + df["station_name"].fillna("")
        + " | "
        + df["station_id"].fillna("")
    )

    return df


def _find_local_meta_path() -> Optional[Path]:
    for p in LOCAL_META_PATHS:
        if p.exists():
            return p
    return None


def load_station_catalog(meta_url: str = META_URL, timeout: int = 30) -> tuple[pd.DataFrame, str]:
    try:
        r = requests.get(meta_url, timeout=timeout)
        r.raise_for_status()
        obj = r.json()
        return _parse_meta_json(obj), "web"
    except Exception:
        local_path = _find_local_meta_path()
        if local_path is None:
            raise FileNotFoundError(
                "Nepodařilo se načíst metadata z webu a lokální hydro_meta1.json nebyl nalezen."
            )
        with open(local_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _parse_meta_json(obj), f"local ({local_path})"


def prepare_station_options(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["station_id", "stream_name", "station_name"]).copy()
    out = out[out["station_id"].astype(str).str.len() > 0].copy()
    out = out[out["stream_name"].astype(str).str.len() > 0].copy()
    out = out[out["station_name"].astype(str).str.len() > 0].copy()
    out = out.sort_values(["stream_name", "station_name", "station_id"]).reset_index(drop=True)
    return out


def get_station_row_by_id(df_meta: pd.DataFrame, station_id: str) -> pd.Series:
    sub = df_meta[df_meta["station_id"] == station_id]
    if sub.empty:
        raise ValueError(f"Stanice {station_id} nebyla nalezena v katalogu.")
    return sub.iloc[0]


def get_recommended_pair_by_label(label: str) -> dict:
    for pair in RECOMMENDED_PAIRS:
        if pair["label"] == label:
            return pair
    raise ValueError(f"Doporučená dvojice '{label}' nebyla nalezena.")


@st.cache_data(ttl=24 * 3600)
def get_station_catalog() -> tuple[pd.DataFrame, str]:
    df_meta, source = load_station_catalog()
    df_meta = prepare_station_options(df_meta)
    return df_meta, source


# =========================================================
# LIVE DATA
# =========================================================
def extract_dt_val_from_tsdata(tsdata, code: str) -> Optional[pd.DataFrame]:
    if not isinstance(tsdata, list) or len(tsdata) == 0:
        return None

    first = tsdata[0]

    if isinstance(first, dict):
        tmp = pd.DataFrame(tsdata)
        dt_col = next((c for c in ["dt", "DT", "dateTime", "time"] if c in tmp.columns), None)
        val_col = next((c for c in ["value", "VAL", code] if c in tmp.columns), None)
        if dt_col is None or val_col is None:
            return None
        return tmp[[dt_col, val_col]].rename(columns={dt_col: "dt", val_col: code})

    if isinstance(first, (list, tuple)):
        n = len(first)
        if n < 2:
            return None
        if n == 2:
            cols = ["dt", "value"]
        elif n == 3:
            cols = ["dt", "value", "flag"]
        else:
            cols = ["dt", "value", "flag", "quality"] + [f"x{i}" for i in range(n - 4)]
        tmp = pd.DataFrame(tsdata, columns=cols[:n])
        return tmp[["dt", "value"]].rename(columns={"dt": "dt", "value": code})

    return None


def extract_ts_from_live_json(obj: dict) -> pd.DataFrame:
    obj_list = obj.get("objList", [])
    if not obj_list:
        return pd.DataFrame()

    first_obj = obj_list[0]
    ts_list = first_obj.get("tsList", [])
    if not ts_list:
        return pd.DataFrame()

    frames = []
    for ts in ts_list:
        code = ts.get("tsConID")
        if code not in ["H", "Q"]:
            continue
        tmp = extract_dt_val_from_tsdata(ts.get("tsData"), code)
        if tmp is not None and not tmp.empty:
            frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = frames[0].copy()
    for tmp in frames[1:]:
        out = out.merge(tmp, on="dt", how="outer")

    out["dt"] = parse_dt_to_local(out["dt"])
    out = out.set_index("dt").sort_index()
    return out.rename(columns={"H": "H_live", "Q": "Q_live"})


@st.cache_data(ttl=300)
def fetch_station_now(station_id: str) -> pd.DataFrame:
    url = f"{BASE_NOW}/{station_id}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    obj = r.json()
    df_live = extract_ts_from_live_json(obj)
    if df_live.empty:
        raise RuntimeError(f"Soubor {url} se stáhl, ale parser z něj nic nevytáhl.")
    return df_live


def build_single_station_features(df_live: pd.DataFrame) -> pd.DataFrame:
    out = df_live.copy()
    out["H"] = pd.to_numeric(out.get("H_live"), errors="coerce")
    out["Q"] = pd.to_numeric(out.get("Q_live"), errors="coerce")
    out["dH_1h"] = out["H"] - out["H"].shift(6)
    out["dH_2h"] = out["H"] - out["H"].shift(12)
    out["rolling_dH_3h"] = out["dH_1h"].rolling(18, min_periods=6).mean()
    return out


def build_dual_station_features(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> pd.DataFrame:
    left = df1.rename(columns={"H_live": f"H_{label1}", "Q_live": f"Q_{label1}"})
    right = df2.rename(columns={"H_live": f"H_{label2}", "Q_live": f"Q_{label2}"})
    df = left.join(right, how="outer").sort_index()

    df[f"H_{label1}"] = pd.to_numeric(df[f"H_{label1}"], errors="coerce")
    df[f"H_{label2}"] = pd.to_numeric(df[f"H_{label2}"], errors="coerce")
    df[f"Q_{label1}"] = pd.to_numeric(df.get(f"Q_{label1}"), errors="coerce")
    df[f"Q_{label2}"] = pd.to_numeric(df.get(f"Q_{label2}"), errors="coerce")

    df[f"dH_{label1}_1h"] = df[f"H_{label1}"] - df[f"H_{label1}"].shift(6)
    df[f"dH_{label2}_1h"] = df[f"H_{label2}"] - df[f"H_{label2}"].shift(6)
    df[f"dH_{label1}_2h"] = df[f"H_{label1}"] - df[f"H_{label1}"].shift(12)
    df[f"dH_{label2}_2h"] = df[f"H_{label2}"] - df[f"H_{label2}"].shift(12)
    df[f"rolling_{label1}_3h"] = df[f"dH_{label1}_1h"].rolling(18, min_periods=6).mean()
    df[f"rolling_{label2}_3h"] = df[f"dH_{label2}_1h"].rolling(18, min_periods=6).mean()
    df["delta_H_2minus1"] = df[f"H_{label2}"] - df[f"H_{label1}"]

    return df


# =========================================================
# HISTORICAL DATA
# =========================================================
def extract_hourly_hh_from_historical_json(obj: dict) -> pd.DataFrame:
    ts_list = obj.get("tsList", [])
    if not ts_list:
        return pd.DataFrame()

    hh_ts = None
    for ts in ts_list:
        if ts.get("tsConID") == "HH":
            hh_ts = ts
            break

    if hh_ts is None:
        return pd.DataFrame()

    ts_data = hh_ts.get("tsData", {}).get("data", {})
    header = ts_data.get("header")
    values = ts_data.get("values", [])

    if not header or not values:
        return pd.DataFrame()

    cols = [c.strip() for c in header.split(",")]
    df = pd.DataFrame(values, columns=cols)

    dt_col = "DT" if "DT" in df.columns else "dt"
    val_col = "VAL" if "VAL" in df.columns else "value"

    df["dt"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce").dt.tz_convert(LOCAL_TZ)
    df["H"] = pd.to_numeric(df[val_col], errors="coerce")

    return df.set_index("dt")[["H"]].sort_index()


@st.cache_data(ttl=24 * 3600)
def fetch_station_historical_hourly(station_id: str, years: list[int]) -> pd.DataFrame:
    frames = []

    for year in years:
        url = f"{BASE_HIST}/H_{station_id}_HQ_{year}.json"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            obj = r.json()
            df_year = extract_hourly_hh_from_historical_json(obj)
            if not df_year.empty:
                frames.append(df_year)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["H"])

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_historical_dual_features(df1_hist: pd.DataFrame, df2_hist: pd.DataFrame) -> pd.DataFrame:
    df = df1_hist.rename(columns={"H": "H_upstream"}).join(
        df2_hist.rename(columns={"H": "H_downstream"}),
        how="inner",
    ).sort_index()

    df["dH_upstream_1h"] = df["H_upstream"] - df["H_upstream"].shift(1)
    df["dH_upstream_2h"] = df["H_upstream"] - df["H_upstream"].shift(2)
    df["rolling_upstream_3h"] = df["dH_upstream_1h"].rolling(3, min_periods=2).mean()

    df["dH_downstream_1h"] = df["H_downstream"] - df["H_downstream"].shift(1)
    df["delta_H_2minus1"] = df["H_downstream"] - df["H_upstream"]

    return df


# =========================================================
# MODELY
# =========================================================
def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def get_pair_model_path(pair_id: str) -> Path:
    return MODELS_DIR / f"{pair_id}.json"


def save_model_to_models_dir(model_json: dict, pair_id: str) -> Path:
    models_dir = ensure_models_dir()
    out_path = models_dir / f"{pair_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)
    return out_path


def load_model_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    coeffs = model.get("coefficients", {})
    return {
        **model,
        "intercept": float(model["intercept"]),
        "coefficients": {k: float(v) for k, v in coeffs.items()},
        "source": str(path),
    }


def load_pair_model_if_exists(pair_id: Optional[str]) -> Optional[dict]:
    if not pair_id:
        return None
    path = get_pair_model_path(pair_id)
    if not path.exists():
        return None
    return load_model_json(path)


def find_default_model_file() -> Optional[Path]:
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path
    return None


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_proba(last_row: pd.Series, model: dict) -> float:
    z = float(model["intercept"])

    if "features" in model and "scaler_mean" in model and "scaler_scale" in model:
        for f in model["features"]:
            x = float(last_row[f])
            mean = float(model["scaler_mean"][f])
            scale = float(model["scaler_scale"][f])
            x_std = (x - mean) / scale if scale != 0 else 0.0
            coef = float(model["coefficients"][f])
            z += coef * x_std
    else:
        for f, coef in model["coefficients"].items():
            if f in last_row.index and pd.notna(last_row[f]):
                z += float(coef) * float(last_row[f])

    return sigmoid(z)


def predict_proba_series(df_dual: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Retrospektivní proba série nad live dual dataframe.
    """
    df = pd.DataFrame(index=df_dual.index)
    df["H_upstream"] = df_dual["H_upstream"]
    df["H_downstream"] = df_dual["H_downstream"]
    df["Q_upstream"] = df_dual.get("Q_upstream")
    df["dH_upstream_1h"] = df_dual["dH_upstream_1h"]
    df["dH_upstream_2h"] = df_dual["dH_upstream_2h"]
    df["rolling_upstream_3h"] = df_dual["rolling_upstream_3h"]

    required = list(set(model.get("features", [])))
    out = df.copy()
    out["proba"] = float("nan")

    if not required:
        return out[["proba"]]

    valid = out.dropna(subset=required).copy()
    if valid.empty:
        return out[["proba"]]

    probs = []
    for _, row in valid.iterrows():
        probs.append(predict_proba(row, model))

    out.loc[valid.index, "proba"] = probs
    return out[["proba"]]


def kayak_decision_layer(H_nesmer: float, proba: float) -> dict:
    H_GO = 100.0
    H_MAYBE = 90.0

    if pd.notna(H_nesmer) and H_nesmer >= H_GO:
        return {"decision": "✅ JEĎ", "reason": "Nesměř je aktuálně na sjízdné hladině.", "eta": "teď"}
    if pd.notna(H_nesmer) and H_MAYBE <= H_nesmer < H_GO:
        if proba >= 0.50:
            return {"decision": "⏳ ZA CHVÍLI", "reason": "Nesměř je hraniční a model čeká další vzestup.", "eta": "1–2 h"}
        return {"decision": "🟡 SLEDUJ", "reason": "Nesměř je hraniční, ale model zatím nevidí silný náběh vlny.", "eta": "nejisté"}
    if proba >= 0.50:
        return {"decision": "⏳ ZA CHVÍLI", "reason": "Teď to ještě nevypadá sjízdně, ale model indikuje pravděpodobný náběh vlny.", "eta": "1–2 h"}
    if proba >= 0.20:
        return {"decision": "🟡 SLEDUJ", "reason": "Zatím nesjízdné, ale možný slabší nebo nejistý náběh vlny.", "eta": "2–3 h / nejisté"}
    return {"decision": "❌ NEJEĎ", "reason": "Nesměř je nízko a model neukazuje významnou vlnu.", "eta": "-"}


def evaluate_oslava_nowcast(df_dual: pd.DataFrame, model: dict) -> dict:
    df = pd.DataFrame(index=df_dual.index)
    df["H_mostiste"] = df_dual["H_upstream"]
    df["H_nesmer"] = df_dual["H_downstream"]
    df["Q_mostiste"] = df_dual.get("Q_upstream")
    df["dH_mostiste_1h"] = df["H_mostiste"] - df["H_mostiste"].shift(6)
    df["dH_mostiste_2h"] = df["H_mostiste"] - df["H_mostiste"].shift(12)
    df["rolling_dH_3h"] = df["dH_mostiste_1h"].rolling(18, min_periods=6).mean()

    required = list(set(["H_mostiste", "H_nesmer", "Q_mostiste"] + model.get("features", [])))
    last = df.dropna(subset=required).iloc[-1]

    proba = predict_proba(last, model)
    hydro_state = "🟢 KLID" if proba < 0.2 else ("🟡 VLNA MOŽNÁ" if proba < 0.5 else "🔴 VLNA PRAVDĚPODOBNÁ")
    kayak = kayak_decision_layer(float(last["H_nesmer"]), proba)

    return {
        "time": last.name,
        "H_mostiste": float(last["H_mostiste"]),
        "H_nesmer": float(last["H_nesmer"]),
        "dH_mostiste_1h": float(last["dH_mostiste_1h"]),
        "dH_mostiste_2h": float(last["dH_mostiste_2h"]),
        "rolling_dH_3h": float(last["rolling_dH_3h"]),
        "proba": float(proba),
        "hydro_state": hydro_state,
        "kayak_decision": kayak["decision"],
        "kayak_reason": kayak["reason"],
        "eta": kayak["eta"],
    }


def evaluate_generic_pair_nowcast(df_dual: pd.DataFrame, model: dict) -> dict:
    df = pd.DataFrame(index=df_dual.index)
    df["H_upstream"] = df_dual["H_upstream"]
    df["H_downstream"] = df_dual["H_downstream"]
    df["Q_upstream"] = df_dual.get("Q_upstream")
    df["dH_upstream_1h"] = df_dual["dH_upstream_1h"]
    df["dH_upstream_2h"] = df_dual["dH_upstream_2h"]
    df["rolling_upstream_3h"] = df_dual["rolling_upstream_3h"]

    required = list(set(model.get("features", [])))
    if not required:
        raise ValueError("Model nemá definované features.")

    last = df.dropna(subset=required).iloc[-1]
    proba = predict_proba(last, model)

    if proba >= 0.75:
        state = "🔴 SILNÝ SIGNÁL"
    elif proba >= 0.50:
        state = "🟡 MOŽNÝ SIGNÁL"
    else:
        state = "🟢 KLID"

    return {
        "time": last.name,
        "proba": float(proba),
        "state": state,
        "H_upstream": float(last["H_upstream"]) if pd.notna(last.get("H_upstream")) else float("nan"),
        "H_downstream": float(last["H_downstream"]) if pd.notna(last.get("H_downstream")) else float("nan"),
    }


def train_pair_model_from_history(
    station1_id: str,
    station2_id: str,
    forecast_h: int = 2,
    rise_thr_cm: float = 2.0,
    years: Optional[list[int]] = None,
) -> tuple[dict, pd.DataFrame]:
    if years is None:
        years = list(range(2020, 2025))

    hist1 = fetch_station_historical_hourly(station1_id, years)
    hist2 = fetch_station_historical_hourly(station2_id, years)

    if hist1.empty or hist2.empty:
        raise ValueError("Nepodařilo se načíst historical data pro jednu nebo obě stanice.")

    df = build_historical_dual_features(hist1, hist2)

    steps = forecast_h
    df["target_rise"] = (
        (df["H_downstream"].shift(-steps) - df["H_downstream"]) >= rise_thr_cm
    ).astype(float)

    features = [
        "H_upstream",
        "dH_upstream_1h",
        "dH_upstream_2h",
        "rolling_upstream_3h",
    ]

    model_df = df.dropna(subset=features + ["target_rise"]).copy()

    if len(model_df) < 200:
        raise ValueError(f"Pro trénink je málo vzorků: {len(model_df)}")

    if model_df["target_rise"].nunique() < 2:
        raise ValueError("Target nemá obě třídy. Zkus změnit rise_thr_cm nebo forecast_h.")

    split_idx = int(len(model_df) * 0.75)
    X_train = model_df[features].iloc[:split_idx]
    X_test = model_df[features].iloc[split_idx:]
    y_train = model_df["target_rise"].iloc[:split_idx]
    y_test = model_df["target_rise"].iloc[split_idx:]

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    test_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)

    model_json = {
        "model_type": "rise_prediction",
        "forecast_h": int(forecast_h),
        "rise_threshold_cm": float(rise_thr_cm),
        "features": features,
        "intercept": float(clf.intercept_[0]),
        "coefficients": {f: float(c) for f, c in zip(features, clf.coef_[0])},
        "threshold": 0.5,
        "watch_threshold": 0.2,
        "auc_test": float(auc),
        "n_samples": int(len(model_df)),
        "class_balance": {
            "0": int((model_df["target_rise"] == 0).sum()),
            "1": int((model_df["target_rise"] == 1).sum()),
        },
        "train_years": years,
        "upstream_id": station1_id,
        "downstream_id": station2_id,
    }

    return model_json, model_df


# =========================================================
# ANALYTIKA
# =========================================================
def estimate_lag_correlation(df: pd.DataFrame, up_col: str, down_col: str, max_lag_h: int = 6) -> pd.DataFrame:
    rows = []
    for lag_steps in range(0, max_lag_h * 6 + 1):
        corr = df[up_col].corr(df[down_col].shift(-lag_steps))
        rows.append(
            {
                "lag_steps_10min": lag_steps,
                "lag_hours": lag_steps / 6,
                "corr": corr,
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# UI
# =========================================================
st.title("🌊 Hydro multiprofile dashboard")
st.caption(
    "Výběr 1 nebo 2 hlásných profilů z aktuálního katalogu ČHMÚ. "
    "Pro Oslavu Mostiště → Nesměř se použije i kajak/model logika."
)
st.info(
    "Dashboard podporuje dva režimy: doporučené hydrologicky smysluplné dvojice "
    "a vlastní experimentální výběr profilů."
)

df_meta, meta_source = get_station_catalog()

with st.sidebar:
    st.header("Výběr profilů")
    st.caption(f"Katalog stanic: {meta_source}")

    selection_mode = st.radio(
        "Režim výběru",
        ["Doporučené dvojice", "Vlastní výběr"],
        index=0,
    )

    station1_row = None
    station2_row = None
    station1_id = None
    station2_id = None
    selected_pair = None
    selected_pair_id = None

    if selection_mode == "Doporučené dvojice":
        pair_labels = [p["label"] for p in RECOMMENDED_PAIRS]
        selected_pair_label = st.selectbox("Doporučená dvojice", pair_labels, index=0)

        selected_pair = get_recommended_pair_by_label(selected_pair_label)
        selected_pair_id = selected_pair.get("pair_id")

        station1_row = get_station_row_by_id(df_meta, selected_pair["upstream_id"])
        station2_row = get_station_row_by_id(df_meta, selected_pair["downstream_id"])

        station1_id = station1_row["station_id"]
        station2_id = station2_row["station_id"]

        st.write("**Řeka:**", selected_pair["stream_name"])
        st.write("**Popis:**", selected_pair["description"])
        st.write("**Profil 1 (upstream):**", station1_row["station_name"])
        st.write("**Profil 2 (downstream):**", station2_row["station_name"])

    else:
        rivers = sorted(df_meta["stream_name"].dropna().unique().tolist())
        selected_river = st.selectbox("Řeka / tok", rivers)

        river_df = df_meta[df_meta["stream_name"] == selected_river].copy()

        station1_label = st.selectbox("První hlásný profil", river_df["label"].tolist(), index=0)
        station2_options = ["(jen jeden profil)"] + river_df["label"].tolist()
        station2_label = st.selectbox("Druhý hlásný profil", station2_options, index=0)

        station1_row = river_df[river_df["label"] == station1_label].iloc[0]
        station1_id = station1_row["station_id"]

        if station2_label == "(jen jeden profil)":
            station2_id = None
            station2_row = None
            selected_pair_id = None
        else:
            station2_row = river_df[river_df["label"] == station2_label].iloc[0]
            station2_id = station2_row["station_id"]
            selected_pair_id = f"custom_{station1_id}_{station2_id}"

        st.write("**Profil 1:**", station1_row["station_name"])
        st.write("**ID 1:**", station1_id)
        if station2_row is not None:
            st.write("**Profil 2:**", station2_row["station_name"])
            st.write("**ID 2:**", station2_id)

    with st.expander("Souřadnice profilů"):
        st.write(
            f"Profil 1: {station1_row['station_name']} | lat={station1_row['lat']}, lon={station1_row['lon']}"
        )
        if station2_row is not None:
            st.write(
                f"Profil 2: {station2_row['station_name']} | lat={station2_row['lat']}, lon={station2_row['lon']}"
            )


try:
    if station2_id is not None and station1_id == station2_id:
        st.error("Profil 1 a profil 2 nesmí být stejná stanice.")
        st.stop()

    if station2_id is None:
        live1 = fetch_station_now(station1_id)
        feat1 = build_single_station_features(live1)

        last = feat1.dropna(subset=["H"]).iloc[-1]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Single-profile režim")
            st.write(f"**Stanice:** {station1_row['station_name']}")
            st.write(f"**Řeka:** {station1_row['stream_name']}")
            st.write(f"**Poslední update:** {last.name}")
        with c2:
            st.metric("H", f"{safe_float(last['H']):.1f} cm")
            q_val = safe_float(last["Q"]) if pd.notna(last.get("Q")) else None
            st.metric("Q", f"{q_val:.3f} m³/s" if q_val is not None and not pd.isna(q_val) else "-")
        with c3:
            dh1 = safe_float(last["dH_1h"]) if pd.notna(last.get("dH_1h")) else float("nan")
            roll = safe_float(last["rolling_dH_3h"]) if pd.notna(last.get("rolling_dH_3h")) else float("nan")
            st.metric("dH / 1 h", f"{dh1:.1f} cm")
            st.metric("rolling dH / 3 h", f"{roll:.2f} cm")

        st.divider()
        st.subheader("Mapa stanice")
        map_df = pd.DataFrame(
            [
                {
                    "station_name": station1_row["station_name"],
                    "latitude": station1_row["lat"],
                    "longitude": station1_row["lon"],
                }
            ]
        )
        st.map(map_df)
        st.dataframe(map_df, width="stretch")

        cutoff = feat1.index.max() - pd.Timedelta("48h")
        st.subheader("Posledních 48 hodin")
        st.line_chart(feat1.loc[feat1.index >= cutoff, ["H"]])

        st.subheader("Trend")
        st.line_chart(feat1.loc[feat1.index >= cutoff, ["dH_1h", "dH_2h", "rolling_dH_3h"]])

        with st.expander("Detail stanice"):
            show_cols = [c for c in ["H", "Q", "dH_1h", "dH_2h", "rolling_dH_3h"] if c in feat1.columns]
            st.dataframe(feat1[show_cols].tail(30), width="stretch")

    else:
        live1 = fetch_station_now(station1_id)
        live2 = fetch_station_now(station2_id)

        df_dual = build_dual_station_features(live1, live2, "upstream", "downstream")
        last_dual = df_dual.dropna(subset=["H_upstream", "H_downstream"]).iloc[-1]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Dual-profile režim")
            st.write(f"**Profil 1:** {station1_row['station_name']}")
            st.write(f"**Profil 2:** {station2_row['station_name']}")
            st.write(f"**Poslední update:** {last_dual.name}")
        with c2:
            st.metric("H profil 1", f"{safe_float(last_dual['H_upstream']):.1f} cm")
            st.metric("H profil 2", f"{safe_float(last_dual['H_downstream']):.1f} cm")
        with c3:
            delta_h = safe_float(last_dual["delta_H_2minus1"]) if pd.notna(last_dual.get("delta_H_2minus1")) else float("nan")
            st.metric("H2 - H1", f"{delta_h:.1f} cm")

        st.divider()
        st.subheader("Mapa vybraných stanic")
        map_rows = [
            {
                "role": "upstream",
                "station_name": station1_row["station_name"],
                "latitude": station1_row["lat"],
                "longitude": station1_row["lon"],
            },
            {
                "role": "downstream",
                "station_name": station2_row["station_name"],
                "latitude": station2_row["lat"],
                "longitude": station2_row["lon"],
            },
        ]
        map_df = pd.DataFrame(map_rows)
        st.map(map_df[["latitude", "longitude"]])
        st.dataframe(map_df, width="stretch")

        selected_model_key = None
        session_model_key = get_session_model_key(selected_pair_id)
        session_model = st.session_state.get(session_model_key) if session_model_key else None
        auto_pair_model = None

        if selection_mode == "Doporučené dvojice" and selected_pair is not None:
            selected_model_key = selected_pair.get("model_key")

        if session_model is not None:
            auto_pair_model = session_model
        elif selected_pair_id is not None:
            auto_pair_model = load_pair_model_if_exists(selected_pair_id)

        st.divider()
        st.subheader("Model pro vybranou dvojici")

        if auto_pair_model is not None:
            model_source_label = "session model" if session_model is not None else f"models/{selected_pair_id}.json"
            st.success(f"Aktivní model: {model_source_label}")
            st.write(f"**Model type:** {auto_pair_model.get('model_type', '-')}")
            st.write(f"**Forecast horizon:** {auto_pair_model.get('forecast_h', '-')}")
            st.write(f"**Tréninkové roky:** {auto_pair_model.get('train_years', '-')}")
        else:
            if selected_pair_id is not None:
                st.info(f"Pro dvojici `{selected_pair_id}` zatím nebyl nalezen uložený model v `models/`.")

        if auto_pair_model is not None:
            try:
                generic_model_result = evaluate_generic_pair_nowcast(df_dual, auto_pair_model)

                g1, g2, g3 = st.columns(3)
                with g1:
                    st.metric("Model state", generic_model_result["state"])
                with g2:
                    st.metric("P(rise)", f"{100 * generic_model_result['proba']:.1f} %")
                with g3:
                    st.metric("Model time", str(generic_model_result["time"]))
            except Exception as e:
                st.warning(f"Automaticky načtený model se nepodařilo vyhodnotit: {e}")

        is_oslava_pair = selected_model_key == "oslava_rise_tplus2h" or (
            station1_id == "0-203-1-471000" and station2_id == "0-203-1-473000"
        )

        if is_oslava_pair:
            model_path = find_default_model_file()
            if model_path is not None:
                try:
                    model = load_model_json(model_path)
                    oslava_result = evaluate_oslava_nowcast(df_dual, model)

                    st.divider()
                    s1, s2, s3 = st.columns([1.7, 1, 1])
                    with s1:
                        st.subheader(oslava_result["kayak_decision"])
                        st.write(oslava_result["kayak_reason"])
                        st.write(f"**ETA:** {oslava_result['eta']}")
                        st.write("---")
                        st.write(f"**Hydro stav:** {oslava_result['hydro_state']}")
                    with s2:
                        st.metric("P(vzestup za 2 h)", f"{100 * oslava_result['proba']:.1f} %")
                        st.metric("H Mostiště", f"{oslava_result['H_mostiste']:.1f} cm")
                    with s3:
                        st.metric("dH Mostiště / 1 h", f"{oslava_result['dH_mostiste_1h']:.1f} cm")
                        st.metric("H Nesměř", f"{oslava_result['H_nesmer']:.1f} cm")
                except Exception as e:
                    st.warning(f"Oslava model se nepodařilo vyhodnotit: {e}")

        st.divider()
        cutoff = df_dual.index.max() - pd.Timedelta("48h")

        st.subheader("Posledních 48 hodin – hladiny")
        st.line_chart(df_dual.loc[df_dual.index >= cutoff, ["H_upstream", "H_downstream"]])

        if auto_pair_model is not None:
            st.subheader("Predikce modelu v čase")
            proba_df = predict_proba_series(df_dual, auto_pair_model)
            proba_recent = proba_df.loc[proba_df.index >= cutoff].copy()

            fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
            ax_pred.plot(proba_recent.index, proba_recent["proba"], label="P(rise)", linewidth=2)

            watch_thr = float(auto_pair_model.get("watch_threshold", 0.2))
            alert_thr = float(auto_pair_model.get("threshold", 0.5))

            ax_pred.axhline(watch_thr, linestyle="--", linewidth=1, label=f"watch ({watch_thr:.2f})")
            ax_pred.axhline(alert_thr, linestyle="--", linewidth=1, label=f"alert ({alert_thr:.2f})")

            ax_pred.set_ylim(0, 1)
            ax_pred.set_ylabel("Probability")
            ax_pred.set_xlabel("Time")
            ax_pred.set_title("Retrospektivní predikce modelu")
            ax_pred.grid(True, alpha=0.3)
            ax_pred.legend()
            st.pyplot(fig_pred)

            st.subheader("Hladiny a predikce dohromady")
            combo = df_dual.join(proba_df, how="left")
            combo_recent = combo.loc[combo.index >= cutoff].copy()

            fig_combo, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

            ax1.plot(combo_recent.index, combo_recent["H_upstream"], label="H upstream")
            ax1.plot(combo_recent.index, combo_recent["H_downstream"], label="H downstream")
            ax1.set_ylabel("H [cm]")
            ax1.set_title("Hladiny")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(combo_recent.index, combo_recent["proba"], label="P(rise)", linewidth=2)
            ax2.axhline(watch_thr, linestyle="--", linewidth=1, label=f"watch ({watch_thr:.2f})")
            ax2.axhline(alert_thr, linestyle="--", linewidth=1, label=f"alert ({alert_thr:.2f})")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Probability")
            ax2.set_xlabel("Time")
            ax2.set_title("Predikce")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            st.pyplot(fig_combo)

        st.subheader("Rozdíl hladin")
        st.line_chart(df_dual.loc[df_dual.index >= cutoff, ["delta_H_2minus1"]])

        st.subheader("Krátkodobé změny")

        plot_df = df_dual.loc[df_dual.index >= cutoff].copy()

        fig_trend, ax_trend = plt.subplots(figsize=(10, 4))

        if "dH_upstream_1h" in plot_df.columns:
            ax_trend.plot(
                plot_df.index,
                plot_df["dH_upstream_1h"],
                label="dH_upstream_1h",
                linewidth=1.8,
            )

        if "dH_downstream_1h" in plot_df.columns:
            ax_trend.plot(
                plot_df.index,
                plot_df["dH_downstream_1h"],
                label="dH_downstream_1h",
                linewidth=1.8,
            )

        if "rolling_upstream_3h" in plot_df.columns:
            ax_trend.plot(
                plot_df.index,
                plot_df["rolling_upstream_3h"],
                label="rolling_upstream_3h",
                color="red",
                linewidth=2.2,
            )

        ax_trend.set_title("Krátkodobé změny")
        ax_trend.set_ylabel("cm")
        ax_trend.set_xlabel("Time")
        ax_trend.grid(True, alpha=0.3)
        ax_trend.legend()

        st.pyplot(fig_trend)

        st.subheader("Lag korelace")
        lag_df = estimate_lag_correlation(df_dual, "H_upstream", "H_downstream", max_lag_h=6)
        best_row = lag_df.loc[lag_df["corr"].idxmax()] if lag_df["corr"].notna().any() else None

        if best_row is not None:
            st.write(f"**Max korelace:** {best_row['corr']:.3f} při lagu ≈ {best_row['lag_hours']:.2f} h")

        st.subheader("Lag korelace")
        lag_df = estimate_lag_correlation(df_dual, "H_upstream", "H_downstream", max_lag_h=6)
        best_row = lag_df.loc[lag_df["corr"].idxmax()] if lag_df["corr"].notna().any() else None

        if best_row is not None:
            st.write(f"**Max korelace:** {best_row['corr']:.3f} při lagu ≈ {best_row['lag_hours']:.2f} h")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lag_df["lag_hours"], lag_df["corr"])
        ax.set_xlabel("Lag [h]")
        ax.set_ylabel("Correlation")
        ax.set_title("Lag korelace profil 1 vs profil 2")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        with st.expander("Detail live dat"):
            show_cols = [
                "H_upstream",
                "Q_upstream",
                "H_downstream",
                "Q_downstream",
                "dH_upstream_1h",
                "dH_downstream_1h",
                "dH_upstream_2h",
                "dH_downstream_2h",
                "rolling_upstream_3h",
                "rolling_downstream_3h",
                "delta_H_2minus1",
            ]
            show_cols = [c for c in show_cols if c in df_dual.columns]
            st.dataframe(df_dual[show_cols].tail(30), width="stretch")

        st.divider()
        st.subheader("Trénink vlastního modelu z historical dat")
        st.caption("Trénink používá historical hourly HQ data (HH) pro zvolené roky.")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            forecast_h = st.number_input("Forecast horizon [h]", min_value=1, max_value=24, value=2, step=1)
        with col_b:
            rise_thr_cm = st.number_input("Target rise threshold [cm]", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
        with col_c:
            year_from = st.number_input("Od roku", min_value=2010, max_value=2025, value=2020, step=1)

        year_to = st.number_input("Do roku", min_value=2010, max_value=2025, value=2024, step=1)

        if year_to < year_from:
            st.error("Koncový rok musí být >= počáteční rok.")
        else:
            train_years = list(range(int(year_from), int(year_to) + 1))
            st.write("**Roky pro trénink:**", train_years)

            train_model = st.button("Natrénovat model z historical dat")

            if train_model:
                try:
                    with st.spinner("Načítám historical data a trénuji model..."):
                        trained_model, model_df = train_pair_model_from_history(
                            station1_id=station1_id,
                            station2_id=station2_id,
                            forecast_h=int(forecast_h),
                            rise_thr_cm=float(rise_thr_cm),
                            years=train_years,
                        )

                    pair_id_for_save = selected_pair_id if selected_pair_id else f"custom_{station1_id}_{station2_id}"

                    trained_model.update(
                        {
                            "pair_id": pair_id_for_save,
                            "upstream_name": station1_row["station_name"],
                            "downstream_name": station2_row["station_name"],
                            "stream_name": station1_row["stream_name"],
                        }
                    )

                    if LOCAL_SAVE_ENABLED:
                        try:
                            saved_path = save_model_to_models_dir(trained_model, pair_id_for_save)
                            st.success(f"Model byl uložen do: {saved_path}")
                        except Exception as e:
                            st.warning(
                                "Model se nepodařilo uložit lokálně do `models/`. "
                                f"Na některých cloudech je filesystem dočasný. Detail: {e}"
                            )

                    session_model_key = get_session_model_key(pair_id_for_save)
                    st.session_state[session_model_key] = trained_model

                    st.success("Model byl natrénován z historical dat.")
                    st.write(f"**Počet vzorků:** {trained_model['n_samples']}")
                    st.write(f"**AUC test:** {trained_model['auc_test']:.3f}")

                    coef_df = pd.Series(trained_model["coefficients"]).to_frame("coef")
                    st.dataframe(coef_df, width="stretch")

                    st.write("### Náhled tréninkových dat")
                    preview_cols = [
                        "H_upstream",
                        "H_downstream",
                        "dH_upstream_1h",
                        "dH_upstream_2h",
                        "rolling_upstream_3h",
                        "target_rise",
                    ]
                    preview_cols = [c for c in preview_cols if c in model_df.columns]
                    st.dataframe(model_df[preview_cols].tail(20), width="stretch")

                    st.download_button(
                        "Stáhnout model JSON",
                        data=json.dumps(trained_model, ensure_ascii=False, indent=2),
                        file_name=f"{pair_id_for_save}.json",
                        mime="application/json",
                    )

                    with st.expander("Model JSON preview"):
                        st.json(trained_model)

                    st.success("Nový model byl aktivován pro aktuální dvojici. Dashboard se obnoví.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Trénink modelu selhal: {e}")

except Exception as e:
    st.error(f"Dashboardu se nepodařilo načíst data: {e}")
    st.stop()