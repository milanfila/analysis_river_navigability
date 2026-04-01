#!/usr/bin/env python3
import json
import os
import smtplib
from email.message import EmailMessage
from typing import Dict, Optional

import pandas as pd
import requests

LOCAL_TZ = "Europe/Prague"
BASE_NOW = "https://opendata.chmi.cz/hydrology/now/data"

STATIONS = {
    "mostiste": "0-203-1-471000",
    "nesmer": "0-203-1-473000",
}


def parse_chmi_dt(series, local_tz: str = LOCAL_TZ) -> pd.DatetimeIndex:
    s = pd.Series(series).astype(str)
    dt = pd.to_datetime(s, utc=True, errors="coerce")
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
        tsdata = ts.get("tsData")
        tmp = extract_dt_val_from_tsdata(tsdata, code)
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

    # Live data jsou po 10 min -> 6 kroků ~ 1 h, 18 kroků ~ 3 h
    out["dH_mostiste_1h"] = out["H_mostiste"] - out["H_mostiste"].shift(6)
    out["rolling_dH_3h"] = out["dH_mostiste_1h"].rolling(18, min_periods=6).mean()
    return out


def read_thresholds() -> Dict[str, float]:
    """
    Kalibrace pro Oslavu pod VD Mostiště:
    - hlavní signál je růst H pod hrází
    - absolutní H je jen podpůrná informace
    """
    return {
        # podpůrná minimální hladina v Mostišti
        "H_min_watch": float(os.getenv("H_MIN_WATCH_CM", "30")),
        "H_min_alert": float(os.getenv("H_MIN_ALERT_CM", "40")),

        # hlavní trigger: růst za 1 h
        "dH_watch": float(os.getenv("DH_WATCH_1H_CM", "3")),
        "dH_alert": float(os.getenv("DH_ALERT_1H_CM", "6")),

        # stabilita růstu
        "rolling_watch": float(os.getenv("ROLLING_WATCH_3H_CM", "1.5")),
        "rolling_alert": float(os.getenv("ROLLING_ALERT_3H_CM", "2.5")),

        # doplňkový průtok pod hrází
        "Q_watch": float(os.getenv("Q_WATCH_M3S", "0.8")),
        "Q_alert": float(os.getenv("Q_ALERT_M3S", "1.2")),

        # orientační ETA vlny do Nesměře
        "ETA_MIN_H": float(os.getenv("ETA_MIN_H", "1.0")),
        "ETA_MAX_H": float(os.getenv("ETA_MAX_H", "2.0")),
    }


def evaluate_alert(live_features: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, object]:
    last = live_features.dropna(subset=["H_mostiste"]).iloc[-1]

    H = float(last["H_mostiste"])
    Hn = float(last["H_nesmer"]) if pd.notna(last.get("H_nesmer")) else float("nan")
    dH1 = float(last["dH_mostiste_1h"]) if pd.notna(last.get("dH_mostiste_1h")) else float("nan")
    trend3 = float(last["rolling_dH_3h"]) if pd.notna(last.get("rolling_dH_3h")) else float("nan")
    Qm = float(last["Q_mostiste"]) if pd.notna(last.get("Q_mostiste")) else float("nan")

    reasons = []
    confidence = 0.0

    # ALERT = jasná manipulační vlna
    alert_main = pd.notna(dH1) and dH1 >= thresholds["dH_alert"]
    alert_support = (
        (H >= thresholds["H_min_alert"]) or
        (pd.notna(trend3) and trend3 >= thresholds["rolling_alert"]) or
        (pd.notna(Qm) and Qm >= thresholds["Q_alert"])
    )

    # WATCH = slabší, ale zajímavý růst
    watch_main = pd.notna(dH1) and dH1 >= thresholds["dH_watch"]
    watch_support = (
        (H >= thresholds["H_min_watch"]) or
        (pd.notna(trend3) and trend3 >= thresholds["rolling_watch"]) or
        (pd.notna(Qm) and Qm >= thresholds["Q_watch"])
    )

    if alert_main and alert_support:
        decision = "ALERT"
        confidence = 0.85

        reasons.append(f"silný růst hladiny pod hrází: dH_mostiste_1h = {dH1:.1f} cm")
        if H >= thresholds["H_min_alert"]:
            reasons.append(f"dostatečná absolutní hladina: H_mostiste = {H:.1f} cm")
        if pd.notna(trend3) and trend3 >= thresholds["rolling_alert"]:
            reasons.append(f"stabilní růst: rolling_dH_3h = {trend3:.1f} cm")
        if pd.notna(Qm) and Qm >= thresholds["Q_alert"]:
            reasons.append(f"zvýšený průtok pod hrází: Q_mostiste = {Qm:.3f} m3/s")

    elif watch_main or (watch_main and watch_support):
        decision = "WATCH"
        confidence = 0.55

        reasons.append(f"pozorovaný růst hladiny: dH_mostiste_1h = {dH1:.1f} cm")
        if H >= thresholds["H_min_watch"]:
            reasons.append(f"podpůrně i hladina: H_mostiste = {H:.1f} cm")
        if pd.notna(trend3) and trend3 >= thresholds["rolling_watch"]:
            reasons.append(f"růst potvrzen trendem: rolling_dH_3h = {trend3:.1f} cm")
        if pd.notna(Qm) and Qm >= thresholds["Q_watch"]:
            reasons.append(f"podpůrně i průtok: Q_mostiste = {Qm:.3f} m3/s")

    else:
        decision = "NO_ALERT"
        confidence = 0.05
        reasons.append("bez významného růstu hladiny pod hrází")

    # orientační interpretace pro Nesměř
    if decision == "ALERT":
        interpretation = (
            f"Pravděpodobná manipulační vlna. "
            f"Odhad dopadu do Nesměře za cca {thresholds['ETA_MIN_H']:.0f}–{thresholds['ETA_MAX_H']:.0f} h."
        )
    elif decision == "WATCH":
        interpretation = (
            f"Možný začátek vlny nebo slabší manipulace. "
            f"Doporučeno dál sledovat Mostiště a Nesměř během následujících {thresholds['ETA_MAX_H']:.0f} h."
        )
    else:
        interpretation = "Aktuálně bez známek významné manipulace nádrže."

    return {
        "time": str(last.name),
        "H_mostiste": H,
        "H_nesmer": Hn,
        "Q_mostiste": Qm,
        "dH_mostiste_1h": dH1,
        "rolling_dH_3h": trend3,
        "decision": decision,
        "confidence": confidence,
        "reasons": reasons,
        "interpretation": interpretation,
        "thresholds": thresholds,
    }


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
    live_features = build_live_features()
    thresholds = read_thresholds()
    result = evaluate_alert(live_features, thresholds)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    always_email = os.getenv("ALWAYS_EMAIL", "false").lower() == "true"

    if result["decision"] == "ALERT":
        subject = "Oslava ALERT – pravděpodobná manipulační vlna"
    elif result["decision"] == "WATCH":
        subject = "Oslava WATCH – možný růst hladiny"
    else:
        subject = "Oslava status – bez významné změny"

    reasons_text = "\n- ".join(result["reasons"]) if result["reasons"] else "žádný trigger"

    body = f"""Čas: {result['time']}

Mostiště:
- H_mostiste: {result['H_mostiste']:.1f} cm
- Q_mostiste: {result['Q_mostiste']:.3f} m3/s
- dH_mostiste_1h: {result['dH_mostiste_1h']:.1f} cm
- rolling_dH_3h: {result['rolling_dH_3h']:.1f} cm

Nesměř:
- H_nesmer: {result['H_nesmer']:.1f} cm

Decision: {result['decision']}
Confidence: {result['confidence']:.2f}

Interpretace:
{result['interpretation']}

Důvody:
- {reasons_text}

Thresholds:
{json.dumps(result['thresholds'], ensure_ascii=False, indent=2)}
"""

    if result["decision"] in {"ALERT", "WATCH"} or always_email:
        send_email(subject, body)
        print("Email sent.")
    else:
        print("No email sent.")


if __name__ == "__main__":
    main()