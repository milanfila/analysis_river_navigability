import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Hydro multiprofile dashboard", page_icon="🌊", layout="wide")

LOCAL_TZ = "Europe/Prague"
BASE_NOW = "https://opendata.chmi.cz/hydrology/now/data"
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
        "lat", "lon",
        "dry_h", "spa1_h", "spa2_h", "spa3_h", "spa4_h",
        "dry_q", "spa1_q", "spa2_q", "spa3_q", "spa4_q",
        "catchment_area_km2",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in [
        "station_id", "station_code", "station_name", "stream_name",
        "spa_type", "stage_unit", "flow_unit", "basin_code"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["label"] = (
        df["stream_name"].fillna("") + " | "
        + df["station_name"].fillna("") + " | "
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
            raise FileNotFoundError("Nepodařilo se načíst metadata z webu a lokální hydro_meta1.json nebyl nalezen.")
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
def get_station_catalog():
    df_meta, source = load_station_catalog()
    df_meta = prepare_station_options(df_meta)
    return df_meta, source


# =========================================================
# LIVE DATA
# =========================================================
def parse_chmi_dt(series, local_tz: str = LOCAL_TZ):
    dt = pd.to_datetime(pd.Series(series).astype(str), utc=True, errors="coerce")
    return pd.DatetimeIndex(dt).tz_convert(local_tz)


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

    out["dt"] = parse_chmi_dt(out["dt"])
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
# MODEL PRO OSLAVU
# =========================================================
def find_default_model_file() -> Optional[Path]:
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path
    return None


def load_model_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    coeffs = model["coefficients"]
    return {
        **model,
        "intercept": float(model["intercept"]),
        "coefficients": {k: float(v) for k, v in coeffs.items()},
        "source": str(path),
    }


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


# =========================================================
# ANALYTIKA
# =========================================================
def estimate_lag_correlation(df: pd.DataFrame, up_col: str, down_col: str, max_lag_h: int = 6) -> pd.DataFrame:
    rows = []
    for lag_steps in range(0, max_lag_h * 6 + 1):
        corr = df[up_col].corr(df[down_col].shift(-lag_steps))
        rows.append({
            "lag_steps_10min": lag_steps,
            "lag_hours": lag_steps / 6,
            "corr": corr,
        })
    return pd.DataFrame(rows)


def train_experimental_pair_model(df_dual: pd.DataFrame, forecast_h: int = 2, rise_thr_cm: float = 2.0) -> dict:
    """
    Experimentální MVP model:
    target = zda downstream za +forecast_h hodin vzroste o alespoň rise_thr_cm.
    Vstupy = upstream H a změny.
    Trénuje se nad dostupnou časovou řadou v df_dual.
    """
    df = df_dual.copy()

    steps = forecast_h * 6  # 10min data
    df["target_rise"] = ((df["H_downstream"].shift(-steps) - df["H_downstream"]) >= rise_thr_cm).astype(float)

    features = [
        "H_upstream",
        "dH_upstream_1h",
        "dH_upstream_2h",
        "rolling_upstream_3h",
    ]

    model_df = df.dropna(subset=features + ["target_rise"]).copy()

    if len(model_df) < 50:
        raise ValueError(f"Pro experimentální trénink je zatím málo vzorků: {len(model_df)}")

    if model_df["target_rise"].nunique() < 2:
        raise ValueError("Target nemá obě třídy. Zkus změnit rise_thr_cm nebo forecast_h.")

    split_idx = int(len(model_df) * 0.75)
    X_train = model_df[features].iloc[:split_idx]
    X_test = model_df[features].iloc[split_idx:]
    y_train = model_df["target_rise"].iloc[:split_idx]
    y_test = model_df["target_rise"].iloc[split_idx:]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    test_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)

    return {
        "model_type": "rise_prediction_experimental",
        "forecast_h": forecast_h,
        "rise_threshold_cm": rise_thr_cm,
        "features": features,
        "intercept": float(clf.intercept_[0]),
        "coefficients": {f: float(c) for f, c in zip(features, clf.coef_[0])},
        "auc_test": float(auc),
        "n_samples": int(len(model_df)),
        "class_balance": {
            "0": int((model_df["target_rise"] == 0).sum()),
            "1": int((model_df["target_rise"] == 1).sum()),
        },
    }


# =========================================================
# UI
# =========================================================
st.title("🌊 Hydro multiprofile dashboard")
st.caption("Výběr 1 nebo 2 hlásných profilů z aktuálního katalogu ČHMÚ. Pro Oslavu Mostiště → Nesměř se použije i kajak/model logika.")
st.info("Dashboard podporuje dva režimy: doporučené hydrologicky smysluplné dvojice a vlastní experimentální výběr profilů.")

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

    if selection_mode == "Doporučené dvojice":
        pair_labels = [p["label"] for p in RECOMMENDED_PAIRS]
        selected_pair_label = st.selectbox("Doporučená dvojice", pair_labels, index=0)

        selected_pair = get_recommended_pair_by_label(selected_pair_label)

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
        else:
            station2_row = river_df[river_df["label"] == station2_label].iloc[0]
            station2_id = station2_row["station_id"]

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

    # -----------------------------------------------------
    # SINGLE PROFILE
    # -----------------------------------------------------
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
            st.metric("H", f"{float(last['H']):.1f} cm")
            q_val = float(last["Q"]) if pd.notna(last.get("Q")) else None
            st.metric("Q", f"{q_val:.3f} m³/s" if q_val is not None else "-")
        with c3:
            dh1 = float(last["dH_1h"]) if pd.notna(last.get("dH_1h")) else float("nan")
            roll = float(last["rolling_dH_3h"]) if pd.notna(last.get("rolling_dH_3h")) else float("nan")
            st.metric("dH / 1 h", f"{dh1:.1f} cm")
            st.metric("rolling dH / 3 h", f"{roll:.2f} cm")

        cutoff = feat1.index.max() - pd.Timedelta("48h")
        st.subheader("Posledních 48 hodin")
        st.line_chart(feat1.loc[feat1.index >= cutoff, ["H"]])

        st.subheader("Trend")
        st.line_chart(feat1.loc[feat1.index >= cutoff, ["dH_1h", "dH_2h", "rolling_dH_3h"]])

        with st.expander("Detail stanice"):
            show_cols = [c for c in ["H", "Q", "dH_1h", "dH_2h", "rolling_dH_3h"] if c in feat1.columns]
            st.dataframe(feat1[show_cols].tail(30), width="stretch")

    # -----------------------------------------------------
    # DUAL PROFILE
    # -----------------------------------------------------
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
            st.metric("H profil 1", f"{float(last_dual['H_upstream']):.1f} cm")
            st.metric("H profil 2", f"{float(last_dual['H_downstream']):.1f} cm")
        with c3:
            delta_h = float(last_dual["delta_H_2minus1"]) if pd.notna(last_dual.get("delta_H_2minus1")) else float("nan")
            st.metric("H2 - H1", f"{delta_h:.1f} cm")

        selected_model_key = None
        if selection_mode == "Doporučené dvojice" and selected_pair is not None:
            selected_model_key = selected_pair.get("model_key")

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
                        st.metric("P(vzestup za 2 h)", f"{100*oslava_result['proba']:.1f} %")
                        st.metric("H Mostiště", f"{oslava_result['H_mostiste']:.1f} cm")
                    with s3:
                        st.metric("dH Mostiště / 1 h", f"{oslava_result['dH_mostiste_1h']:.1f} cm")
                        st.metric("H Nesměř", f"{oslava_result['H_nesmer']:.1f} cm")
                except Exception as e:
                    st.warning(f"Oslava model se nepodařilo vyhodnotit: {e}")

        st.divider()
        cutoff = df_dual.index.max() - pd.Timedelta("48h")

        st.subheader("Posledních 48 hodin – hladiny")
        st.line_chart(
            df_dual.loc[df_dual.index >= cutoff, ["H_upstream", "H_downstream"]]
        )

        st.subheader("Rozdíl hladin")
        st.line_chart(df_dual.loc[df_dual.index >= cutoff, ["delta_H_2minus1"]])

        st.subheader("Krátkodobé změny")
        trend_cols = [c for c in ["dH_upstream_1h", "dH_downstream_1h"] if c in df_dual.columns]
        if trend_cols:
            st.line_chart(df_dual.loc[df_dual.index >= cutoff, trend_cols])

        st.subheader("Lag korelace")
        lag_df = estimate_lag_correlation(df_dual, "H_upstream", "H_downstream", max_lag_h=6)
        best_row = lag_df.loc[lag_df["corr"].idxmax()] if lag_df["corr"].notna().any() else None

        if best_row is not None:
            st.write(
                f"**Max korelace:** {best_row['corr']:.3f} při lagu ≈ {best_row['lag_hours']:.2f} h"
            )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lag_df["lag_hours"], lag_df["corr"])
        ax.set_xlabel("Lag [h]")
        ax.set_ylabel("Correlation")
        ax.set_title("Lag korelace profil 1 vs profil 2")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        with st.expander("Detail dat"):
            show_cols = [
                "H_upstream", "Q_upstream",
                "H_downstream", "Q_downstream",
                "dH_upstream_1h", "dH_downstream_1h",
                "dH_upstream_2h", "dH_downstream_2h",
                "rolling_upstream_3h", "rolling_downstream_3h",
                "delta_H_2minus1",
            ]
            show_cols = [c for c in show_cols if c in df_dual.columns]
            st.dataframe(df_dual[show_cols].tail(30), width="stretch")

        st.divider()
        st.subheader("Experimentální trénink vlastního modelu")
        st.caption("Tahle verze je MVP. Trénuje jednoduchý logistický model nad dostupnou časovou řadou v dashboardu. Pro plnohodnotný model je vhodné napojit historická data.")

        col_a, col_b = st.columns(2)
        with col_a:
            forecast_h = st.number_input("Forecast horizon [h]", min_value=1, max_value=12, value=2, step=1)
        with col_b:
            rise_thr_cm = st.number_input("Target rise threshold [cm]", min_value=0.5, max_value=20.0, value=2.0, step=0.5)

        train_model = st.button("Natrénovat model pro zvolenou dvojici")

        if train_model:
            try:
                trained_model = train_experimental_pair_model(
                    df_dual=df_dual,
                    forecast_h=int(forecast_h),
                    rise_thr_cm=float(rise_thr_cm),
                )

                st.success("Experimentální model byl natrénován.")
                st.write(f"**Počet vzorků:** {trained_model['n_samples']}")
                st.write(f"**AUC test:** {trained_model['auc_test']:.3f}")

                coef_df = pd.Series(trained_model["coefficients"]).to_frame("coef")
                st.dataframe(coef_df, width="stretch")

                model_export = {
                    **trained_model,
                    "upstream_id": station1_id,
                    "downstream_id": station2_id,
                    "upstream_name": station1_row["station_name"],
                    "downstream_name": station2_row["station_name"],
                    "stream_name": station1_row["stream_name"],
                }

                st.download_button(
                    "Stáhnout model JSON",
                    data=json.dumps(model_export, ensure_ascii=False, indent=2),
                    file_name=f"model_{station1_id}_{station2_id}_tplus{forecast_h}h.json",
                    mime="application/json",
                )

                with st.expander("Model JSON preview"):
                    st.json(model_export)

            except Exception as e:
                st.error(f"Trénink modelu selhal: {e}")

except Exception as e:
    st.error(f"Dashboardu se nepodařilo načíst data: {e}")