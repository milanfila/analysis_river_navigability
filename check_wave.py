import json
import math
import os
import smtplib
import subprocess
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import requests

BASE_NOW = "https://opendata.chmi.cz/hydrology/now/data"
LOCAL_TZ = "Europe/Prague"

STATIONS = {
    "mostiste": "0-203-1-471000",
    "nesmer": "0-203-1-473000",
}

MODEL_PATH = Path("basic_dashboard/oslava_model_rise_tplus2h.json")
STATE_PATH = Path("alert_state.json")
HISTORY_PATH = Path("alert_history.csv")


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return float(default)
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    return int(raw)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_chmi_dt(series):
    dt = pd.to_datetime(pd.Series(series).astype(str), utc=True, errors="coerce")
    return pd.DatetimeIndex(dt).tz_convert(LOCAL_TZ)


def extract_dt_val_from_tsdata(tsdata, code: str):
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


def fetch_station_now(station_id: str) -> pd.DataFrame:
    url = f"{BASE_NOW}/{station_id}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    obj = r.json()
    df_live = extract_ts_from_live_json(obj)
    if df_live.empty:
        raise RuntimeError(f"Soubor {url} se stáhl, ale parser z něj nic nevytáhl.")
    return df_live


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

    out["dH_mostiste_1h"] = out["H_mostiste"] - out["H_mostiste"].shift(6)
    out["dH_mostiste_2h"] = out["H_mostiste"] - out["H_mostiste"].shift(12)
    out["rolling_dH_3h"] = out["dH_mostiste_1h"].rolling(18, min_periods=6).mean()

    return out


def load_model(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_proba(last_row: pd.Series, model: dict) -> float:
    z = float(model["intercept"])

    for f in model["features"]:
        x = float(last_row[f])
        mean = float(model["scaler_mean"][f])
        scale = float(model["scaler_scale"][f])
        x_std = (x - mean) / scale if scale != 0 else 0.0
        coef = float(model["coefficients"][f])
        z += coef * x_std

    return sigmoid(z)


def build_decision(last_row: pd.Series, proba: float, model: dict) -> dict:
    watch_thr = env_float("MODEL_WATCH_THRESHOLD", float(model.get("threshold", 0.2)))
    alert_thr = env_float("MODEL_ALERT_THRESHOLD", 0.5)

    H_nesmer = float(last_row["H_nesmer"]) if pd.notna(last_row.get("H_nesmer")) else float("nan")
    H_mostiste = float(last_row["H_mostiste"]) if pd.notna(last_row.get("H_mostiste")) else float("nan")
    dH1 = float(last_row["dH_mostiste_1h"]) if pd.notna(last_row.get("dH_mostiste_1h")) else float("nan")
    dH2 = float(last_row["dH_mostiste_2h"]) if pd.notna(last_row.get("dH_mostiste_2h")) else float("nan")
    roll3 = float(last_row["rolling_dH_3h"]) if pd.notna(last_row.get("rolling_dH_3h")) else float("nan")

    if H_nesmer >= 100:
        kayak_decision = "JEĎ"
        eta = "teď"
        reason = "Nesměř je aktuálně na sjízdné hladině."
    elif proba >= alert_thr:
        kayak_decision = "ZA CHVÍLI"
        eta = "1–2 h"
        reason = "Model indikuje pravděpodobný náběh vlny."
    elif proba >= watch_thr:
        kayak_decision = "SLEDUJ"
        eta = "2–3 h / nejisté"
        reason = "Možný slabší nebo nejistý náběh vlny."
    else:
        kayak_decision = "NEJEĎ"
        eta = "-"
        reason = "Aktuálně bez známek významné vlny."

    if proba >= alert_thr:
        alert_level = "ALERT"
    elif proba >= watch_thr:
        alert_level = "WATCH"
    else:
        alert_level = "NO_ALERT"

    return {
        "time": str(last_row.name),
        "H_mostiste": H_mostiste,
        "H_nesmer": H_nesmer,
        "dH_mostiste_1h": dH1,
        "dH_mostiste_2h": dH2,
        "rolling_dH_3h": roll3,
        "proba": proba,
        "alert_level": alert_level,
        "kayak_decision": kayak_decision,
        "eta": eta,
        "reason": reason,
        "watch_threshold": watch_thr,
        "alert_threshold": alert_thr,
    }


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "last_alert_level": "NO_ALERT",
            "last_sent_at": None,
            "last_proba": 0.0,
        }

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def hours_since(iso_ts: str | None) -> float:
    if not iso_ts:
        return 1e9
    then = datetime.fromisoformat(iso_ts)
    now = datetime.now(timezone.utc)
    return (now - then).total_seconds() / 3600.0


def should_send_alert(result: dict, state: dict) -> tuple[bool, str]:
    cooldown_h = env_int("ALERT_COOLDOWN_HOURS", 6)
    proba_jump = env_float("ALERT_PROBA_JUMP", 0.20)

    prev_level = state.get("last_alert_level", "NO_ALERT")
    prev_proba = float(state.get("last_proba", 0.0))
    last_sent_at = state.get("last_sent_at")

    current_level = result["alert_level"]
    current_proba = float(result["proba"])

    if current_level == "NO_ALERT":
        return False, "no alert state"

    if current_level != prev_level:
        return True, f"level change {prev_level} -> {current_level}"

    if abs(current_proba - prev_proba) >= proba_jump:
        return True, f"probability jump {prev_proba:.2f} -> {current_proba:.2f}"

    if hours_since(last_sent_at) >= cooldown_h:
        return True, f"cooldown passed ({cooldown_h} h)"

    return False, "suppressed by anti-spam"


def append_history_row(result: dict, email_sent: bool, anti_spam_reason: str) -> None:
    row = {
        "time": result["time"],
        "alert_level": result["alert_level"],
        "kayak_decision": result["kayak_decision"],
        "eta": result["eta"],
        "proba": result["proba"],
        "H_mostiste": result["H_mostiste"],
        "H_nesmer": result["H_nesmer"],
        "dH_mostiste_1h": result["dH_mostiste_1h"],
        "dH_mostiste_2h": result["dH_mostiste_2h"],
        "rolling_dH_3h": result["rolling_dH_3h"],
        "email_sent": int(email_sent),
        "anti_spam_reason": anti_spam_reason,
    }

    df_row = pd.DataFrame([row])

    if HISTORY_PATH.exists():
        df_old = pd.read_csv(HISTORY_PATH)
        df_all = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_all = df_row

    df_all = df_all.drop_duplicates(subset=["time"], keep="last")
    df_all = df_all.sort_values("time").tail(5000)
    df_all.to_csv(HISTORY_PATH, index=False)


def commit_runtime_files() -> None:
    if os.getenv("GITHUB_ACTIONS") != "true":
        return

    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        check=True,
    )

    subprocess.run(["git", "add", str(STATE_PATH), str(HISTORY_PATH)], check=True)

    diff_check = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)

    if diff_check.returncode == 0:
        print("No state/history changes to commit.")
        return

    subprocess.run(["git", "commit", "-m", "Update alert state/history [skip ci]"], check=True)
    subprocess.run(["git", "push"], check=True)


def send_email(subject: str, body: str) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]
    mail_to = os.environ["MAIL_TO"]
    mail_from = os.getenv("MAIL_FROM", smtp_user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    state = load_state()

    live_features = build_live_features()
    required = list(set(["H_mostiste", "H_nesmer"] + model["features"]))
    last_row = live_features.dropna(subset=required).iloc[-1]

    proba = predict_proba(last_row, model)
    result = build_decision(last_row, proba, model)

    send_it, send_reason = should_send_alert(result, state)
    result["anti_spam_decision"] = send_reason

    print(json.dumps(result, ensure_ascii=False, indent=2))

    body = f"""Oslava wave alert

Čas: {result['time']}
Rozhodnutí: {result['kayak_decision']}
Alert level: {result['alert_level']}
ETA: {result['eta']}

Pravděpodobnost vzestupu hladiny za 2 h: {100*result['proba']:.1f} %

H Mostiště: {result['H_mostiste']:.1f} cm
H Nesměř: {result['H_nesmer']:.1f} cm
dH Mostiště / 1 h: {result['dH_mostiste_1h']:.1f} cm
dH Mostiště / 2 h: {result['dH_mostiste_2h']:.1f} cm
rolling dH / 3 h: {result['rolling_dH_3h']:.2f} cm

Důvod:
{result['reason']}

Anti-spam:
{send_reason}
"""

    email_sent = False

    if send_it:
        subject = f"Oslava {result['alert_level']} – {result['kayak_decision']}"
        send_email(subject, body)
        print("Email sent.")
        state["last_sent_at"] = utc_now_iso()
        email_sent = True
    else:
        print("No email sent.")

    state["last_alert_level"] = result["alert_level"]
    state["last_proba"] = float(result["proba"])

    save_state(state)
    append_history_row(result, email_sent=email_sent, anti_spam_reason=send_reason)
    commit_runtime_files()


if __name__ == "__main__":
    main()