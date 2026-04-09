import math
from typing import Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Oslava kajak dashboard", page_icon="🚣", layout="wide")

LOCAL_TZ = "Europe/Prague"
BASE_NOW = "https://opendata.chmi.cz/hydrology/now/data"

STATIONS = {
    "mostiste": "0-203-1-471000",
    "nesmer": "0-203-1-473000",
}

DEFAULT_MODEL = {
    "intercept": -8.0,
    "coefficients": {
        "H_mostiste": 0.08,
        "dH_mostiste_1h": 0.85,
        "rolling_dH_3h": 0.35,
    },
}


def parse_chmi_dt(series, local_tz: str = LOCAL_TZ) -> pd.DatetimeIndex:
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


@st.cache_data(ttl=300)
def build_live_features() -> pd.DataFrame:
    live_most = fetch_station_now(STATIONS["mostiste"]).rename(
        columns={"H_live": "H_mostiste_live", "Q_live": "Q_mostiste_live"}
    )
    live_nesm = fetch_station_now(STATIONS["nesmer"]).rename(
        columns={"H_live": "H_nesmer_live", "Q_live": "Q_nesmer_live"}
    )

    live_df = live_most.join(live_nesm, how="outer", lsuffix="_most", rsuffix="_nesm").sort_index()

    out = live_df.copy()
    out["H_mostiste"] = pd.to_numeric(out["H_mostiste_live"], errors="coerce")
    out["H_nesmer"] = pd.to_numeric(out["H_nesmer_live"], errors="coerce")
    out["Q_mostiste"] = pd.to_numeric(out.get("Q_mostiste_live"), errors="coerce")
    out["Q_nesmer"] = pd.to_numeric(out.get("Q_nesmer_live"), errors="coerce")

    out["dH_mostiste_1h"] = out["H_mostiste"] - out["H_mostiste"].shift(6)
    out["rolling_dH_3h"] = out["dH_mostiste_1h"].rolling(18, min_periods=6).mean()
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_proba(last_row: pd.Series, model: dict) -> float:
    z = float(model["intercept"])
    z += float(model["coefficients"]["H_mostiste"]) * float(last_row["H_mostiste"])
    z += float(model["coefficients"]["dH_mostiste_1h"]) * float(last_row["dH_mostiste_1h"])
    z += float(model["coefficients"]["rolling_dH_3h"]) * float(last_row["rolling_dH_3h"])
    return sigmoid(z)


def evaluate_nowcast(df: pd.DataFrame, model: dict) -> dict:
    last = df.dropna(subset=["H_mostiste", "dH_mostiste_1h", "rolling_dH_3h"]).iloc[-1]
    proba = predict_proba(last, model)

    if proba >= 0.75:
        decision = "🟢 JEĎ"
        explanation = "Pravděpodobná manipulační vlna."
    elif proba >= 0.45:
        decision = "🟡 SLEDUJ"
        explanation = "Možný začátek vlny nebo slabší manipulace."
    else:
        decision = "🔴 NEJEĎ"
        explanation = "Bez známek významné manipulace nádrže."

    return {
        "time": last.name,
        "H_mostiste": float(last["H_mostiste"]),
        "H_nesmer": float(last["H_nesmer"]) if pd.notna(last.get("H_nesmer")) else float("nan"),
        "Q_mostiste": float(last["Q_mostiste"]) if pd.notna(last.get("Q_mostiste")) else float("nan"),
        "dH_mostiste_1h": float(last["dH_mostiste_1h"]),
        "rolling_dH_3h": float(last["rolling_dH_3h"]),
        "proba_tplus_2h": float(proba),
        "decision": decision,
        "explanation": explanation,
    }


st.title("🚣 Oslava kajak dashboard")
st.caption("Minimalistická live verze: Mostiště → Nesměř, nowcast sjízdnosti za cca 2 hodiny.")

with st.sidebar:
    st.header("Model")
    intercept = st.number_input("Intercept", value=float(DEFAULT_MODEL["intercept"]), step=0.1)
    coef_H = st.number_input("Coef H_mostiste", value=float(DEFAULT_MODEL["coefficients"]["H_mostiste"]), step=0.01)
    coef_dH = st.number_input("Coef dH_mostiste_1h", value=float(DEFAULT_MODEL["coefficients"]["dH_mostiste_1h"]), step=0.01)
    coef_roll = st.number_input("Coef rolling_dH_3h", value=float(DEFAULT_MODEL["coefficients"]["rolling_dH_3h"]), step=0.01)
    st.info("Tyhle defaulty jsou ukázkové. Až budeš chtít, nahradíš je koeficienty z tvého natrénovaného modelu.")

model = {
    "intercept": intercept,
    "coefficients": {
        "H_mostiste": coef_H,
        "dH_mostiste_1h": coef_dH,
        "rolling_dH_3h": coef_roll,
    },
}

try:
    live_features = build_live_features()
    result = evaluate_nowcast(live_features, model)

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        st.subheader(result["decision"])
        st.write(result["explanation"])
        st.write(f"**Poslední update:** {result['time']}")
    with c2:
        st.metric("P(sjízdné za 2 h)", f"{100*result['proba_tplus_2h']:.1f} %")
        st.metric("H Mostiště", f"{result['H_mostiste']:.1f} cm")
    with c3:
        st.metric("dH Mostiště / 1 h", f"{result['dH_mostiste_1h']:.1f} cm")
        st.metric("H Nesměř", f"{result['H_nesmer']:.1f} cm")

    st.divider()

    st.subheader("Posledních 48 hodin")
    recent = live_features[["H_mostiste", "H_nesmer"]].last("48h")
    st.line_chart(recent)

    st.subheader("Trend pod hrází")
    recent_dh = live_features[["dH_mostiste_1h", "rolling_dH_3h"]].last("48h")
    st.line_chart(recent_dh)

    with st.expander("Poslední řádky dat"):
        st.dataframe(
            live_features[["H_mostiste", "Q_mostiste", "dH_mostiste_1h", "rolling_dH_3h", "H_nesmer"]].tail(20),
            use_container_width=True,
        )

except Exception as e:
    st.error(f"Dashboardu se nepodařilo načíst live data: {e}")
    st.stop()
