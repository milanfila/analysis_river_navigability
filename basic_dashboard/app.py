import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Hydro multiprofile dashboard", page_icon="🌊", layout="wide")

LOCAL_TZ = "Europe/Prague"
BASE_NOW = "https://opendata.chmi.cz/hydrology/now/data"
META_URL = "https://opendata.chmi.cz/hydrology/historical/metadata/meta1.json"

# fallback na lokální metadata
LOCAL_META_PATHS = [
    Path("hydro_meta1.json"),
    Path("../hydro_meta1.json"),
]

DEFAULT_MODEL_PATHS = [
    Path("oslava_model_rise_tplus2h.json"),
    Path("models/oslava_model_rise_tplus2h.json"),
    Path("basic_dashboard/oslava_model_rise_tplus2h.json"),
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
    # mapování do původních názvů features pro Mostiště/Nesměř
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
    """
    Jednoduchá lag korelace pro 10min data.
    max_lag_h=6 znamená test zpoždění 0..6 hodin.
    """
    rows = []
    for lag_steps in range(0, max_lag_h * 6 + 1):
        corr = df[up_col].corr(df[down_col].shift(-lag_steps))
        rows.append({
            "lag_steps_10min": lag_steps,
            "lag_hours": lag_steps / 6,
            "corr": corr,
        })
    out = pd.DataFrame(rows)
    return out


# =========================================================
# UI
# =========================================================
st.title("🌊 Hydro multiprofile dashboard")
st.caption("Výběr 1 nebo 2 hlásných profilů z aktuálního katalogu ČHMÚ. Pro Oslavu Mostiště → Nesměř se použije i kajak/model logika.")

df_meta, meta_source = get_station_catalog()

with st.sidebar:
    st.header("Výběr profilů")
    st.caption(f"Katalog stanic: {meta_source}")

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

    st.divider()
    st.write("**Profil 1:**", station1_row["station_name"])
    st.write("**ID 1:**", station1_id)
    if station2_row is not None:
        st.write("**Profil 2:**", station2_row["station_name"])
        st.write("**ID 2:**", station2_id)

    auto_refresh = st.toggle("Auto refresh přes cache TTL", value=True)

try:
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

        # Oslava speciální logika: Mostiště -> Nesměř
        is_oslava_pair = (
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
                "delta_H_2minus1",
            ]
            show_cols = [c for c in show_cols if c in df_dual.columns]
            st.dataframe(df_dual[show_cols].tail(30), width="stretch")

except Exception as e:
    st.error(f"Dashboardu se nepodařilo načíst data: {e}")