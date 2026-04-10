import json
import math
from pathlib import Path
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

DEFAULT_MODEL_PATHS = [
    Path("oslava_model_rise_tplus2h.json"),
    Path("models/oslava_model_rise_tplus2h.json"),
    Path("oslava_model_tplus2h.json"),
    Path("models/oslava_model_tplus2h.json"),
]


def load_model_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    required_top = {"intercept", "coefficients"}
    if not required_top.issubset(model.keys()):
        raise ValueError(f"Model JSON {path} nemá požadované klíče: {required_top}")

    coeffs = model["coefficients"]
    return {
        **model,
        "intercept": float(model["intercept"]),
        "coefficients": {k: float(v) for k, v in coeffs.items()},
        "source": str(path),
    }


def find_model_file() -> Path | None:
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path
    return None


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

    # 10min data
    out["dH_mostiste_1h"] = out["H_mostiste"] - out["H_mostiste"].shift(6)
    out["dH_mostiste_2h"] = out["H_mostiste"] - out["H_mostiste"].shift(12)
    out["rolling_dH_3h"] = out["dH_mostiste_1h"].rolling(18, min_periods=6).mean()
    return out


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


def kayak_decision_layer(result: dict) -> dict:
    """
    Praktická kajak logika:
    - aktuální sjízdnost podle H_nesmer
    - forecast podle modelové pravděpodobnosti vzestupu
    """
    Hn = result["H_nesmer"]
    proba = result["proba_tplus_2h"]

    # pracovní prahy, lze později doladit
    H_GO = 100.0
    H_MAYBE = 90.0

    if pd.notna(Hn) and Hn >= H_GO:
        return {
            "kayak_decision": "✅ JEĎ",
            "kayak_reason": "Nesměř je aktuálně na sjízdné hladině.",
            "eta": "teď",
        }

    if pd.notna(Hn) and H_MAYBE <= Hn < H_GO:
        if proba >= 0.50:
            return {
                "kayak_decision": "⏳ ZA CHVÍLI",
                "kayak_reason": "Nesměř je hraniční a model čeká další vzestup.",
                "eta": "1–2 h",
            }
        return {
            "kayak_decision": "🟡 SLEDUJ",
            "kayak_reason": "Nesměř je hraniční, ale model zatím nevidí silný náběh vlny.",
            "eta": "nejisté",
        }

    # H_nesmer < 90
    if proba >= 0.50:
        return {
            "kayak_decision": "⏳ ZA CHVÍLI",
            "kayak_reason": "Teď to ještě nevypadá sjízdně, ale model indikuje pravděpodobný náběh vlny.",
            "eta": "1–2 h",
        }
    if proba >= 0.20:
        return {
            "kayak_decision": "🟡 SLEDUJ",
            "kayak_reason": "Zatím nesjízdné, ale možný slabší nebo nejistý náběh vlny.",
            "eta": "2–3 h / nejisté",
        }
    return {
        "kayak_decision": "❌ NEJEĎ",
        "kayak_reason": "Nesměř je nízko a model neukazuje významnou vlnu.",
        "eta": "-",
    }


def evaluate_nowcast(df: pd.DataFrame, model: dict) -> dict:
    if "features" in model:
        required = list(set(["H_mostiste", "H_nesmer", "Q_mostiste"] + model["features"]))
    else:
        required = ["H_mostiste", "H_nesmer", "Q_mostiste", "dH_mostiste_1h", "rolling_dH_3h"]

    last = df.dropna(subset=required).iloc[-1]
    proba = predict_proba(last, model)

    if model.get("model_type") == "rise_prediction":
        threshold = float(model.get("threshold", 0.2))
        if proba < threshold:
            decision = "🟢 KLID"
            explanation = "Bez známek významného vzestupu hladiny v Nesměři během následujících 2 hodin."
        elif proba < 0.5:
            decision = "🟡 VLNA MOŽNÁ"
            explanation = "Možný náběh vlny v Nesměři během následujících 2 hodin."
        else:
            decision = "🔴 VLNA PRAVDĚPODOBNÁ"
            explanation = "Model indikuje pravděpodobný vzestup hladiny v Nesměři během následujících 2 hodin."
        metric_label = "P(vzestup hladiny za 2 h)"
    else:
        if proba >= 0.75:
            decision = "🟢 JEĎ"
            explanation = "Pravděpodobná manipulační vlna."
        elif proba >= 0.45:
            decision = "🟡 SLEDUJ"
            explanation = "Možný začátek vlny nebo slabší manipulace."
        else:
            decision = "🔴 NEJEĎ"
            explanation = "Bez známek významné manipulace nádrže."
        metric_label = "P(sjízdné za 2 h)"

    if pd.notna(last.get("rolling_dH_3h")) and float(last["rolling_dH_3h"]) > 2:
        hydro_note = "Stabilní růst hladiny na Mostišti."
    elif pd.notna(last.get("dH_mostiste_1h")) and float(last["dH_mostiste_1h"]) > 3:
        hydro_note = "Rychlá manipulace nádrže."
    else:
        hydro_note = "Bez výrazné změny odtoku."

    base_result = {
        "H_nesmer": float(last["H_nesmer"]) if pd.notna(last.get("H_nesmer")) else float("nan"),
        "proba_tplus_2h": float(proba),
    }
    kayak = kayak_decision_layer(base_result)

    return {
        "time": last.name,
        "H_mostiste": float(last["H_mostiste"]),
        "H_nesmer": float(last["H_nesmer"]) if pd.notna(last.get("H_nesmer")) else float("nan"),
        "Q_mostiste": float(last["Q_mostiste"]) if pd.notna(last.get("Q_mostiste")) else float("nan"),
        "dH_mostiste_1h": float(last["dH_mostiste_1h"]) if pd.notna(last.get("dH_mostiste_1h")) else float("nan"),
        "dH_mostiste_2h": float(last["dH_mostiste_2h"]) if pd.notna(last.get("dH_mostiste_2h")) else float("nan"),
        "rolling_dH_3h": float(last["rolling_dH_3h"]) if pd.notna(last.get("rolling_dH_3h")) else float("nan"),
        "proba_tplus_2h": float(proba),
        "decision": decision,
        "explanation": explanation,
        "hydro_note": hydro_note,
        "metric_label": metric_label,
        "model_type": model.get("model_type", "legacy"),
        "kayak_decision": kayak["kayak_decision"],
        "kayak_reason": kayak["kayak_reason"],
        "eta": kayak["eta"],
    }


st.title("🚣 Oslava kajak dashboard")
st.caption("Kajak mode: Mostiště → Nesměř. Dashboard kombinuje aktuální stav v Nesměři a model pravděpodobnosti vzestupu hladiny.")

with st.sidebar:
    st.header("Model")

    detected_model_path = find_model_file()
    use_json_model = st.toggle("Použít JSON model", value=detected_model_path is not None)

    if detected_model_path is not None:
        st.caption(f"Nalezený model: `{detected_model_path}`")
    else:
        st.caption("JSON model nebyl automaticky nalezen.")

    model_path_text = st.text_input(
        "Cesta k modelu JSON",
        value=str(detected_model_path) if detected_model_path is not None else "oslava_model_rise_tplus2h.json",
    )

    loaded_model = None
    if use_json_model:
        try:
            loaded_model = load_model_json(Path(model_path_text))
            st.success(f"Načten JSON model: {loaded_model['source']}")
        except Exception as e:
            st.error(f"JSON model se nepodařilo načíst: {e}")

    manual_default = loaded_model if loaded_model is not None else DEFAULT_MODEL

    intercept = st.number_input("Intercept", value=float(manual_default["intercept"]), step=0.1)
    coeff_keys = list(manual_default["coefficients"].keys())
    coef_inputs = {}
    for key in coeff_keys:
        coef_inputs[key] = st.number_input(f"Coef {key}", value=float(manual_default["coefficients"][key]), step=0.01)

    if loaded_model is None:
        st.info("Když JSON model není načtený, dashboard používá ručně zadané koeficienty.")

model = {
    **(loaded_model if loaded_model is not None else {}),
    "intercept": intercept,
    "coefficients": coef_inputs,
}

try:
    live_features = build_live_features()
    result = evaluate_nowcast(live_features, model)

    c1, c2, c3 = st.columns([1.7, 1, 1])
    with c1:
        st.subheader(result["kayak_decision"])
        st.write(result["kayak_reason"])
        st.write(f"**ETA:** {result['eta']}")
        st.write("---")
        st.write(f"**Hydro stav:** {result['decision']}")
        st.write(result["explanation"])
        st.write(result["hydro_note"])
        st.write(f"**Poslední update:** {result['time']}")
        model_source = model_path_text if use_json_model else "ruční koeficienty"
        st.write(f"**Model:** {model_source}")

    with c2:
        st.metric(result["metric_label"], f"{100*result['proba_tplus_2h']:.1f} %")
        st.metric("H Mostiště", f"{result['H_mostiste']:.1f} cm")

    with c3:
        st.metric("dH Mostiště / 1 h", f"{result['dH_mostiste_1h']:.1f} cm")
        st.metric("H Nesměř", f"{result['H_nesmer']:.1f} cm")
        st.metric("ETA", result["eta"])

    st.divider()

    if st.button("🔄 Obnovit data"):
        st.cache_data.clear()
        st.rerun()

    cutoff = live_features.index.max() - pd.Timedelta("48h")

    st.subheader("Posledních 48 hodin")
    recent = live_features.loc[live_features.index >= cutoff, ["H_mostiste", "H_nesmer"]]
    st.line_chart(recent)

    st.subheader("Trend pod hrází")
    recent_dh = live_features.loc[
        live_features.index >= cutoff,
        ["dH_mostiste_1h", "dH_mostiste_2h", "rolling_dH_3h"]
    ]
    st.line_chart(recent_dh)

    with st.expander("Poslední řádky dat"):
        st.dataframe(
            live_features[[
                "H_mostiste",
                "Q_mostiste",
                "dH_mostiste_1h",
                "dH_mostiste_2h",
                "rolling_dH_3h",
                "H_nesmer",
            ]].tail(20),
            width="stretch",
        )

except Exception as e:
    st.error(f"Dashboardu se nepodařilo načíst live data: {e}")
    st.stop()