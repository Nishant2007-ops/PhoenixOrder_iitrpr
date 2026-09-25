"""
PHOENIX ORDER — Predictive Cyber Defense System
SIH26153 // World Model // Network Intelligence

Analyst command-center dashboard for PhoenixOrder's Delta-LSTM behavioral
prediction pipeline. This file is a VISUALIZATION LAYER ONLY.

It does not train, retrain, or run the LSTM. It reads the already-generated
`stage_predictions.csv` produced by the existing pipeline
(src/analysis/build_attack_timeline.py -> add_defense_recommendations.py)
and renders it as an operational review interface.

Run with:
    streamlit run dashboard.py

Data contract (read defensively — columns may be absent, and this file
must never invent values for a missing column):
    session, segment_id, sequence_index, current_timestamp, next_timestamp,
    behavior_score, significant_feature_count, behavior_evidence,
    stage, score, confidence, evidence,
    mitre_tactic_id, mitre_tactic, mitre_techniques, defense_recommendations
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# CONSTANTS
# ============================================================================

DATASETS = {
    "CSE-CIC-IDS2018": Path(__file__).resolve().parent.parent / "stage_predictions.csv",
    "CTU-13": Path(__file__).resolve().parent.parent / "ctu13_predictions.csv",
}

# Columns the dashboard knows how to use. Nothing outside this list is ever
# fabricated; anything not present in the CSV is simply not shown.
KNOWN_COLUMNS = [
    "session", "segment_id", "sequence_index", "current_timestamp",
    "next_timestamp", "behavior_score", "significant_feature_count",
    "behavior_evidence", "stage", "score", "confidence", "evidence",
    "mitre_tactic_id", "mitre_tactic", "mitre_techniques",
    "defense_recommendations",
]

UNKNOWN_STAGE_LABELS = {"unknown", "unclassified", "none", "nan", ""}

COLOR = {
    "bg": "#0a0d10",
    "panel": "#11161a",
    "panel_alt": "#151b20",
    "border": "#232b31",
    "text": "#c9d3d8",
    "text_dim": "#6b7880",
    "text_faint": "#4a545b",
    "cyan": "#4fd1e8",
    "teal": "#3a8f8a",
    "amber": "#e0a44d",
    "red": "#d9534f",
    "green": "#4fae7a",
}

STAGE_COLOR = {
    "critical": COLOR["red"],
    "high": COLOR["amber"],
    "elevated": COLOR["amber"],
    "unclassified": COLOR["text_dim"],
    "unknown": COLOR["text_dim"],
}

# ============================================================================
# PAGE CONFIG + CSS
# ============================================================================

def configure_page() -> None:
    st.set_page_config(
        page_title="Phoenix Order_iitrpr",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Mono', monospace;
        }}

        .stApp {{
            background:
                linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px),
                {COLOR["bg"]};
            background-size: 28px 28px, 28px 28px, auto;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}
        .block-container {{
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1600px;
        }}

        section[data-testid="stSidebar"] {{
            background: {COLOR["panel"]};
            border-right: 1px solid {COLOR["border"]};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}

        h1, h2, h3, h4, h5 {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {COLOR["text"]} !important;
            letter-spacing: 0.04em;
        }}

        p, span, div, label {{
            color: {COLOR["text"]};
        }}

        /* generic panel */
        .po-panel {{
            background: {COLOR["panel"]};
            border: 1px solid {COLOR["border"]};
            border-radius: 3px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }}
        .po-panel-title {{
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.12em;
            color: {COLOR["text_dim"]};
            text-transform: uppercase;
            border-bottom: 1px solid {COLOR["border"]};
            padding-bottom: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
        }}

        /* header */
        .po-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid {COLOR["border"]};
            padding-bottom: 12px;
            margin-bottom: 14px;
        }}
        .po-logo {{
            font-size: 21px;
            font-weight: 700;
            letter-spacing: 0.14em;
            color: {COLOR["text"]};
        }}
        .po-logo span {{ color: {COLOR["cyan"]}; }}
        .po-subtitle {{
            font-size: 10.5px;
            letter-spacing: 0.1em;
            color: {COLOR["text_dim"]};
            margin-top: 1px;
        }}
        .po-meta {{
            font-size: 9.5px;
            letter-spacing: 0.08em;
            color: {COLOR["text_faint"]};
            margin-top: 3px;
        }}
        .po-header-right {{
            text-align: right;
            display: flex;
            gap: 26px;
        }}
        .po-status-block {{ text-align: right; }}
        .po-status-label {{
            font-size: 9.5px;
            color: {COLOR["text_faint"]};
            letter-spacing: 0.1em;
        }}
        .po-status-value {{
            font-size: 12.5px;
            font-weight: 600;
            color: {COLOR["text"]};
            letter-spacing: 0.06em;
        }}
        .dot {{
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            margin-right: 5px;
            box-shadow: 0 0 6px currentColor;
        }}
        .dot-online {{ background: {COLOR["green"]}; color: {COLOR["green"]}; }}
        .dot-offline {{ background: {COLOR["red"]}; color: {COLOR["red"]}; }}

        /* KPI strip */
        .po-kpi-row {{ display: flex; gap: 0; margin-bottom: 14px; }}
        .po-kpi {{
            flex: 1;
            background: {COLOR["panel"]};
            border: 1px solid {COLOR["border"]};
            padding: 9px 14px;
            border-right: none;
        }}
        .po-kpi:last-child {{ border-right: 1px solid {COLOR["border"]}; }}
        .po-kpi-label {{
            font-size: 9px;
            letter-spacing: 0.1em;
            color: {COLOR["text_faint"]};
            margin-bottom: 3px;
        }}
        .po-kpi-value {{
            font-size: 15px;
            font-weight: 600;
            color: {COLOR["text"]};
            letter-spacing: 0.02em;
        }}

        /* forecast rows */
        .po-state-row {{ display: flex; justify-content: space-between; align-items: center; }}
        .po-state-box {{
            border: 1px solid {COLOR["border"]};
            background: {COLOR["panel_alt"]};
            padding: 10px 14px;
            flex: 1;
        }}
        .po-state-label {{ font-size: 9.5px; color: {COLOR["text_faint"]}; letter-spacing: 0.08em; }}
        .po-state-ts {{ font-size: 13px; color: {COLOR["cyan"]}; font-weight: 600; margin-top: 2px; }}
        .po-arrow {{ color: {COLOR["text_faint"]}; font-size: 18px; padding: 0 14px; }}

        .po-delta-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 2px;
            border-bottom: 1px solid {COLOR["border"]};
            font-size: 12px;
        }}
        .po-delta-row:last-child {{ border-bottom: none; }}
        .po-delta-name {{ color: {COLOR["text_dim"]}; letter-spacing: 0.03em; }}
        .po-delta-bar-wrap {{ flex: 1; margin: 0 12px; background: {COLOR["panel_alt"]}; height: 6px; align-self: center; position: relative; }}
        .po-delta-bar {{ height: 6px; position: absolute; top: 0; }}
        .po-delta-val {{ width: 64px; text-align: right; font-weight: 600; }}
        .po-up {{ color: {COLOR["cyan"]}; }}
        .po-down {{ color: {COLOR["amber"]}; }}
        .po-flat {{ color: {COLOR["text_dim"]}; }}

        /* behavior score */
        .po-score-wrap {{ text-align: center; padding: 6px 0 2px 0; }}
        .po-score-num {{
            font-size: 52px;
            font-weight: 700;
            color: {COLOR["cyan"]};
            text-shadow: 0 0 24px rgba(79,209,232,0.25);
            line-height: 1;
        }}
        .po-score-label {{
            font-size: 10px;
            letter-spacing: 0.12em;
            color: {COLOR["text_dim"]};
            margin-top: 6px;
        }}
        .po-score-sub {{
            font-size: 11px;
            color: {COLOR["text_faint"]};
            margin-top: 10px;
            letter-spacing: 0.05em;
        }}
        .po-evidence-item {{
            font-size: 12px;
            padding: 5px 0;
            border-bottom: 1px solid {COLOR["border"]};
            display: flex;
            justify-content: space-between;
        }}
        .po-evidence-item:last-child {{ border-bottom: none; }}

        /* threat context */
        .po-stage-badge {{
            display: inline-block;
            padding: 5px 12px;
            border: 1px solid;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.1em;
        }}
        .po-field {{ margin-top: 10px; }}
        .po-field-label {{ font-size: 9.5px; color: {COLOR["text_faint"]}; letter-spacing: 0.1em; }}
        .po-field-value {{ font-size: 12.5px; color: {COLOR["text"]}; margin-top: 2px; }}
        .po-field-muted {{ font-size: 12px; color: {COLOR["text_faint"]}; font-style: italic; margin-top: 2px; }}

        /* defense */
        .po-rec-item {{
            display: flex;
            gap: 8px;
            padding: 7px 0;
            border-bottom: 1px solid {COLOR["border"]};
            font-size: 12.5px;
            color: {COLOR["text"]};
        }}
        .po-rec-item:last-child {{ border-bottom: none; }}
        .po-rec-arrow {{ color: {COLOR["cyan"]}; flex-shrink: 0; }}

        /* event stream */
        .po-event {{
            display: flex;
            gap: 12px;
            padding: 6px 0;
            border-bottom: 1px solid {COLOR["border"]};
            font-size: 11.5px;
        }}
        .po-event:last-child {{ border-bottom: none; }}
        .po-event-ts {{ color: {COLOR["text_faint"]}; width: 68px; flex-shrink: 0; }}
        .po-event-txt {{ color: {COLOR["text_dim"]}; }}

        .po-tag {{
            display: inline-block;
            font-size: 9px;
            letter-spacing: 0.08em;
            padding: 2px 7px;
            border: 1px solid {COLOR["border"]};
            color: {COLOR["text_faint"]};
            margin-left: 8px;
        }}

        /* streamlit widget re-skin */
        div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label,
        div[data-testid="stMultiSelect"] label {{
            font-size: 10.5px !important;
            letter-spacing: 0.08em;
            color: {COLOR["text_dim"]} !important;
            text-transform: uppercase;
        }}
        .stButton button {{
            background: {COLOR["panel_alt"]};
            border: 1px solid {COLOR["border"]};
            color: {COLOR["text"]};
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.06em;
            border-radius: 2px;
        }}
        .stButton button:hover {{
            border-color: {COLOR["cyan"]};
            color: {COLOR["cyan"]};
        }}
        hr {{ border-color: {COLOR["border"]}; }}

        .po-section-label {{
            font-size: 10.5px;
            font-weight: 600;
            letter-spacing: 0.14em;
            color: {COLOR["text_dim"]};
            text-transform: uppercase;
            margin: 18px 0 8px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid {COLOR["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame | None:
    """Load stage_predictions.csv once. Returns None if the file is absent
    or unreadable — the caller is responsible for showing an honest error
    panel rather than crashing or fabricating rows."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty:
        return df

    for ts_col in ("current_timestamp", "next_timestamp"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    return df


def has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def get(row: pd.Series, col: str, default: Any = None) -> Any:
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val


# ============================================================================
# PARSING HELPERS (evidence / recommendations / mitre techniques may be
# stored as stringified lists/dicts, delimited text, or plain strings —
# handle all of these without inventing content)
# ============================================================================

def parse_structured(raw: Any) -> list:
    """Best-effort parse of a cell that may be a JSON/py-literal list of
    dicts, a plain delimited string, or a single string. Never raises."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if not isinstance(raw, str) or not raw.strip():
        return []
    s = raw.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            val = parser(s)
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, dict):
                return [val]
        except Exception:
            continue
    # fall back to line/semicolon/pipe delimited plain text
    for delim in ("\n", ";", "|"):
        if delim in s:
            return [p.strip() for p in s.split(delim) if p.strip()]
    return [s]


def evidence_to_rows(raw: Any) -> list[dict]:
    """Normalize behavior_evidence / evidence into {feature, pct, direction}
    rows regardless of source format."""
    items = parse_structured(raw)
    rows = []
    for it in items:
        if isinstance(it, dict):
            name = it.get("feature") or it.get("name") or it.get("key")
            pct = it.get("pct_change") or it.get("change") or it.get("delta") or it.get("value")
            if name is None:
                continue
            try:
                pct = float(pct)
            except (TypeError, ValueError):
                pct = None
            rows.append({"feature": str(name), "pct": pct})
        else:
            text = str(it)
            rows.append({"feature": text, "pct": None})
    return rows


def recommendations_to_list(raw: Any) -> list[str]:
    items = parse_structured(raw)
    out = []
    for it in items:
        if isinstance(it, dict):
            text = it.get("recommendation") or it.get("action") or it.get("text") or json.dumps(it)
        else:
            text = str(it)
        text = text.strip()
        if text:
            out.append(text)
    return out


def techniques_to_list(raw: Any) -> list[str]:
    items = parse_structured(raw)
    out = []
    for it in items:
        if isinstance(it, dict):
            tid = it.get("id") or it.get("technique_id") or ""
            name = it.get("name") or it.get("technique") or ""
            out.append(f"{tid} {name}".strip())
        else:
            out.append(str(it))
    return out


def is_unknown_stage(stage_val: Any) -> bool:
    if stage_val is None:
        return True
    return str(stage_val).strip().lower() in UNKNOWN_STAGE_LABELS


def stage_color(stage_val: Any) -> str:
    if is_unknown_stage(stage_val):
        return COLOR["text_dim"]
    key = str(stage_val).strip().lower()
    return STAGE_COLOR.get(key, COLOR["cyan"])


def fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "—"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def fmt_ts(ts: Any) -> str:
    if ts is None:
        return "—"
    if isinstance(ts, str):
        return ts
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


# ============================================================================
# RENDER: ERROR PANEL
# ============================================================================

def render_error_panel(reason: str) -> None:
    st.markdown(
        f"""
        <div class="po-panel" style="border-color:{COLOR['red']}; margin-top:40px;">
            <div style="color:{COLOR['red']}; font-size:15px; font-weight:700; letter-spacing:0.08em;">
                ⚠ PREDICTION DATA UNAVAILABLE
            </div>
            <div style="margin-top:10px; color:{COLOR['text_dim']}; font-size:12.5px; line-height:1.7;">
                {reason}<br><br>
                Run the prediction pipeline before launching the dashboard, then place
                <code>stage_predictions.csv</code> in the working directory
                (or alongside <code>dashboard.py</code>).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# RENDER: HEADER
# ============================================================================

def render_header() -> None:
    st.markdown(
        f"""
        <div class="po-header">
            <div>
                <div class="po-logo">PHOENIX<span>◈</span>ORDER_IITRPR</div>
                <div class="po-subtitle">PREDICTIVE CYBER DEFENSE SYSTEM</div>
                <div class="po-meta">SIH26153 // WORLD MODEL // NETWORK INTELLIGENCE</div>
            </div>
            <div class="po-header-right">
                <div class="po-status-block">
                    <div class="po-status-label">DATA MODE</div>
                    <div class="po-status-value" style="color:{COLOR['amber']};">HISTORICAL REVIEW</div>
                </div>
                <div class="po-status-block">
                    <div class="po-status-label">MODEL</div>
                    <div class="po-status-value">DELTA-LSTM</div>
                </div>
                <div class="po-status-block">
                    <div class="po-status-label">SYSTEM STATUS</div>
                    <div class="po-status-value"><span class="dot dot-online"></span>ONLINE</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# RENDER: SIDEBAR (CONTROLS) — filters actually drive the data below
# ============================================================================

def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    st.sidebar.markdown(
        f"<div style='font-size:11px; font-weight:700; letter-spacing:0.12em; "
        f"color:{COLOR['cyan']}; margin-bottom:14px;'>◈ COMMAND</div>",
        unsafe_allow_html=True,
    )

    filtered = df.copy()

    # --- DATASET / SESSION / SEGMENT ---
    st.sidebar.markdown("**DATASET**")
    if has(df, "session"):
        sessions = ["ALL"] + sorted(df["session"].dropna().astype(str).unique().tolist())
        sel_session = st.sidebar.selectbox("SESSION", sessions, key="sel_session")
        if sel_session != "ALL":
            filtered = filtered[filtered["session"].astype(str) == sel_session]
    else:
        st.sidebar.caption("session — not in dataset")

    if has(df, "segment_id"):
        segments = ["ALL"] + sorted(filtered["segment_id"].dropna().astype(str).unique().tolist())
        sel_segment = st.sidebar.selectbox("SEGMENT", segments, key="sel_segment")
        if sel_segment != "ALL":
            filtered = filtered[filtered["segment_id"].astype(str) == sel_segment]

    # --- MODEL INFO (static, informational — not editable, nothing to fake) ---
    st.sidebar.markdown("**MODEL**")
    st.sidebar.caption("MODEL TYPE · Delta-LSTM World Model")
    st.sidebar.caption("CONTEXT WINDOW · per pipeline config")
    st.sidebar.caption("FORECAST HORIZON · next timestep")

    # --- FILTERS ---
    st.sidebar.markdown("**FILTERS**")
    if has(df, "behavior_score") and filtered["behavior_score"].notna().any():
        bmin = float(df["behavior_score"].min())
        bmax = float(df["behavior_score"].max())
        if bmin < bmax:
            thresh = st.sidebar.slider("BEHAVIOR SCORE THRESHOLD", bmin, bmax, bmin, key="sel_bscore")
            filtered = filtered[filtered["behavior_score"] >= thresh]

    if has(df, "significant_feature_count") and filtered["significant_feature_count"].notna().any():
        fmax = int(df["significant_feature_count"].max())
        if fmax > 0:
            fmin_sel = st.sidebar.slider("MIN SIGNIFICANT FEATURES", 0, fmax, 0, key="sel_sigfeat")
            filtered = filtered[filtered["significant_feature_count"] >= fmin_sel]

    if has(df, "stage"):
        stages = ["ALL"] + sorted(df["stage"].fillna("Unknown").astype(str).unique().tolist())
        sel_stage = st.sidebar.selectbox("STAGE", stages, key="sel_stage")
        if sel_stage != "ALL":
            filtered = filtered[filtered["stage"].fillna("Unknown").astype(str) == sel_stage]

    if has(df, "mitre_tactic"):
        tactics = ["ALL"] + sorted(df["mitre_tactic"].dropna().astype(str).unique().tolist())
        if len(tactics) > 1:
            sel_tactic = st.sidebar.selectbox("MITRE TACTIC", tactics, key="sel_tactic")
            if sel_tactic != "ALL":
                filtered = filtered[filtered["mitre_tactic"].astype(str) == sel_tactic]

    if filtered.empty:
        st.sidebar.warning("No rows match current filters.")
        return filtered, None

    # --- TIME / SEQUENCE SELECTOR (drives the active prediction) ---
    st.sidebar.markdown("**TIMELINE REPLAY**")
    filtered = filtered.reset_index(drop=True)
    n = len(filtered)

    if "replay_idx" not in st.session_state:
        st.session_state.replay_idx = 0
    st.session_state.replay_idx = min(st.session_state.replay_idx, n - 1)

    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("◀", width="stretch"):
        st.session_state.replay_idx = max(0, st.session_state.replay_idx - 1)
    if c2.button("▶ AUTO", width="stretch"):
        st.session_state.autoplay = not st.session_state.get("autoplay", False)
    if c3.button("▶", width="stretch"):
        st.session_state.replay_idx = min(n - 1, st.session_state.replay_idx + 1)

    idx = st.sidebar.slider("SEQUENCE INDEX", 0, max(n - 1, 0), st.session_state.replay_idx, key="seq_slider")
    st.session_state.replay_idx = idx

    if st.session_state.get("autoplay", False) and n > 1:
        st.sidebar.caption("◉ AUTOPLAY ACTIVE — replaying stored predictions")
        time.sleep(0.9)
        st.session_state.replay_idx = (st.session_state.replay_idx + 1) % n
        st.rerun()

    selected_row = filtered.iloc[st.session_state.replay_idx]
    return filtered, selected_row


# ============================================================================
# RENDER: KPI STRIP
# ============================================================================

def render_kpi_strip(row: pd.Series | None, total_rows: int) -> None:
    if row is None:
        return

    stage_val = get(row, "stage", "UNCLASSIFIED")
    stage_display = "UNCLASSIFIED" if is_unknown_stage(stage_val) else str(stage_val).upper()

    kpis = [
        ("SESSION", str(get(row, "session", "—"))),
        ("SEGMENT", str(get(row, "segment_id", "—"))),
        ("SEQ INDEX", str(get(row, "sequence_index", "—"))),
        ("BEHAVIOR SCORE", f"{get(row, 'behavior_score', 0):.3f}" if get(row, "behavior_score") is not None else "—"),
        ("SIGNALS", str(int(get(row, "significant_feature_count", 0))) if get(row, "significant_feature_count") is not None else "—"),
        ("STAGE", stage_display),
    ]

    html = "<div class='po-kpi-row'>"
    for label, val in kpis:
        html += f"""<div class="po-kpi">
            <div class="po-kpi-label">{label}</div>
            <div class="po-kpi-value">{val}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# RENDER: FORECAST PANEL (Section 1)
# ============================================================================

def render_forecast_panel(row: pd.Series) -> None:
    st.markdown('<div class="po-panel">', unsafe_allow_html=True)
    st.markdown(
        "<div class='po-panel-title'><span>NETWORK STATE FORECAST</span>"
        "<span>WORLD MODEL PROJECTION</span></div>",
        unsafe_allow_html=True,
    )

    cur_ts = fmt_ts(get(row, "current_timestamp"))
    nxt_ts = fmt_ts(get(row, "next_timestamp"))

    st.markdown(
        f"""
        <div class="po-state-row">
            <div class="po-state-box">
                <div class="po-state-label">CURRENT STATE</div>
                <div class="po-state-ts">{cur_ts}</div>
            </div>
            <div class="po-arrow">→</div>
            <div class="po-state-box">
                <div class="po-state-label">PREDICTED NEXT STATE</div>
                <div class="po-state-ts">{nxt_ts}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence_rows = evidence_to_rows(get(row, "behavior_evidence"))
    if not evidence_rows:
        evidence_rows = evidence_to_rows(get(row, "evidence"))

    if evidence_rows:
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        max_abs = max((abs(r["pct"]) for r in evidence_rows if r["pct"] is not None), default=1) or 1
        rows_html = ""
        for r in evidence_rows[:8]:
            pct = r["pct"]
            if pct is None:
                arrow, cls, width = "→", "po-flat", 0
            elif pct > 0:
                arrow, cls, width = "↑", "po-up", min(100, abs(pct) / max_abs * 100)
            elif pct < 0:
                arrow, cls, width = "↓", "po-down", min(100, abs(pct) / max_abs * 100)
            else:
                arrow, cls, width = "→", "po-flat", 0
            bar_color = COLOR["cyan"] if cls == "po-up" else (COLOR["amber"] if cls == "po-down" else COLOR["text_faint"])
            fname = r["feature"].replace("_", " ").upper()
            rows_html += f"""
            <div class="po-delta-row">
                <div class="po-delta-name">{arrow} {fname}</div>
                <div class="po-delta-bar-wrap"><div class="po-delta-bar" style="width:{width}%; background:{bar_color};"></div></div>
                <div class="po-delta-val {cls}">{fmt_pct(pct)}</div>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='po-field-muted'>No behavioral evidence fields available for this row.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# RENDER: BEHAVIOR PANEL (Section 2) — core innovation, visually dominant
# ============================================================================

def render_behavior_panel(row: pd.Series) -> None:
    st.markdown('<div class="po-panel" style="border-color:#1c454a;">', unsafe_allow_html=True)
    st.markdown(
        "<div class='po-panel-title'><span>BEHAVIORAL CHANGE ANALYSIS</span>"
        "<span>NOT AN ATTACK PROBABILITY</span></div>",
        unsafe_allow_html=True,
    )

    score = get(row, "behavior_score")
    sig_count = get(row, "significant_feature_count")

    score_str = f"{score:.3f}" if score is not None else "—"
    sig_str = f"{int(sig_count)} SIGNIFICANT CHANGE{'S' if sig_count != 1 else ''}" if sig_count is not None else "NO SIGNAL COUNT AVAILABLE"

    st.markdown(
        f"""
        <div class="po-score-wrap">
            <div class="po-score-num">{score_str}</div>
            <div class="po-score-label">PREDICTED BEHAVIORAL CHANGE SCORE</div>
            <div class="po-score-sub">{sig_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence_rows = evidence_to_rows(get(row, "behavior_evidence"))
    if not evidence_rows:
        evidence_rows = evidence_to_rows(get(row, "evidence"))

    if evidence_rows:
        items_html = ""
        for r in evidence_rows[:6]:
            pct = r["pct"]
            arrow = "↑" if (pct or 0) > 0 else ("↓" if (pct or 0) < 0 else "→")
            cls = "po-up" if (pct or 0) > 0 else ("po-down" if (pct or 0) < 0 else "po-flat")
            fname = r["feature"].replace("_", " ")
            items_html += f"""<div class="po-evidence-item">
                <span>{arrow} {fname}</span><span class="{cls}">{fmt_pct(pct)}</span>
            </div>"""
        st.markdown(f"<div style='margin-top:14px;'>{items_html}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='po-field-muted' style='margin-top:12px;'>"
        f"This score reflects the magnitude of predicted change between the current and "
        f"next network state — it is a measure of state deviation, not a probability of attack."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# RENDER: TIMELINE (Section 3)
# ============================================================================

def render_timeline(df: pd.DataFrame, selected_row: pd.Series | None) -> None:
    st.markdown('<div class="po-panel">', unsafe_allow_html=True)
    st.markdown(
        "<div class='po-panel-title'><span>TEMPORAL BEHAVIOR TIMELINE</span>"
        "<span>OFFLINE ANALYSIS — HISTORICAL PREDICTION REVIEW</span></div>",
        unsafe_allow_html=True,
    )

    if not (has(df, "current_timestamp") and has(df, "behavior_score")):
        st.markdown(
            "<div class='po-field-muted'>Timeline requires current_timestamp and "
            "behavior_score columns, which are not both present.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    plot_df = df.dropna(subset=["current_timestamp", "behavior_score"]).sort_values("current_timestamp")
    if plot_df.empty:
        st.markdown("<div class='po-field-muted'>No plottable rows.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    threshold = plot_df["behavior_score"].quantile(0.75)
    colors = [COLOR["amber"] if v >= threshold else COLOR["cyan"] for v in plot_df["behavior_score"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["current_timestamp"], y=plot_df["behavior_score"],
        mode="lines", line=dict(color=COLOR["teal"], width=1.4),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["current_timestamp"], y=plot_df["behavior_score"],
        mode="markers",
        marker=dict(color=colors, size=6, line=dict(width=1, color=COLOR["bg"])),
        customdata=plot_df[["significant_feature_count"]].values if has(df, "significant_feature_count") else None,
        hovertemplate="TS %{x}<br>SCORE %{y:.3f}<extra></extra>",
        showlegend=False,
        name="behavior_score",
    ))

    if selected_row is not None and get(selected_row, "current_timestamp") is not None:
        sel_ts = selected_row["current_timestamp"]
        sel_score = get(selected_row, "behavior_score")
        if sel_score is not None:
            fig.add_trace(go.Scatter(
                x=[sel_ts], y=[sel_score], mode="markers",
                marker=dict(color=COLOR["red"], size=13, symbol="circle-open", line=dict(width=2, color=COLOR["red"])),
                hoverinfo="skip", showlegend=False,
            ))

    fig.add_hline(y=threshold, line_dash="dot", line_color=COLOR["text_faint"], line_width=1,
                   annotation_text="ELEVATED THRESHOLD (P75)", annotation_font_size=9,
                   annotation_font_color=COLOR["text_faint"])

    fig.update_layout(
        height=290,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=COLOR["panel_alt"],
        paper_bgcolor=COLOR["panel"],
        font=dict(family="IBM Plex Mono, monospace", color=COLOR["text_dim"], size=10),
        xaxis=dict(gridcolor=COLOR["border"], showline=True, linecolor=COLOR["border"], title=None),
        yaxis=dict(gridcolor=COLOR["border"], showline=True, linecolor=COLOR["border"], title="BEHAVIOR SCORE"),
        hoverlabel=dict(bgcolor=COLOR["panel_alt"], font_family="IBM Plex Mono, monospace", font_size=11,
                         bordercolor=COLOR["border"]),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# RENDER: THREAT CONTEXT (Section 4)
# ============================================================================

def render_threat_context(row: pd.Series) -> None:
    st.markdown('<div class="po-panel">', unsafe_allow_html=True)
    st.markdown(
        "<div class='po-panel-title'><span>THREAT CONTEXT</span>"
        "<span>MITRE ATT&CK</span></div>",
        unsafe_allow_html=True,
    )

    stage_val = get(row, "stage")
    stage_display = "UNCLASSIFIED" if is_unknown_stage(stage_val) else str(stage_val).upper()
    badge_color = stage_color(stage_val)

    st.markdown(
        f"""
        <div>
            <div class="po-field-label">STAGE</div>
            <div class="po-stage-badge" style="color:{badge_color}; border-color:{badge_color};">{stage_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_unknown_stage(stage_val):
        st.markdown(
            f"<div class='po-field-muted' style='margin-top:8px;'>"
            f"The attack-stage mapper applies a conservative classification policy — "
            f"an unclassified result reflects insufficient evidence for a stage label, "
            f"not a system failure.</div>",
            unsafe_allow_html=True,
        )

    conf = get(row, "confidence")
    if conf is not None:
        try:
            conf_str = f"{float(conf):.2f}"
        except (TypeError, ValueError):
            conf_str = str(conf)
        st.markdown(
            f"<div class='po-field'><div class='po-field-label'>CONFIDENCE</div>"
            f"<div class='po-field-value'>{conf_str}</div></div>",
            unsafe_allow_html=True,
        )

    tactic = get(row, "mitre_tactic")
    tactic_id = get(row, "mitre_tactic_id")
    if tactic is not None:
        tactic_str = f"{tactic_id} · {tactic}" if tactic_id else str(tactic)
        st.markdown(
            f"<div class='po-field'><div class='po-field-label'>MITRE TACTIC</div>"
            f"<div class='po-field-value'>{tactic_str}</div></div>",
            unsafe_allow_html=True,
        )

    techniques = techniques_to_list(get(row, "mitre_techniques"))
    if techniques:
        tech_html = "".join(f"<div class='po-field-value'>· {t}</div>" for t in techniques[:6])
        st.markdown(
            f"<div class='po-field'><div class='po-field-label'>MITRE TECHNIQUES</div>{tech_html}</div>",
            unsafe_allow_html=True,
        )

    if tactic is None and not techniques:
        st.markdown(
            f"<div class='po-field'><div class='po-field-label'>ATT&CK MAPPING</div>"
            f"<div class='po-field-muted'>NO ATT&CK MAPPING GENERATED</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# RENDER: DEFENSE PANEL (Section 5)
# ============================================================================

def render_defense_panel(row: pd.Series) -> None:
    st.markdown('<div class="po-panel">', unsafe_allow_html=True)
    st.markdown(
        "<div class='po-panel-title'><span>DEFENSIVE RESPONSE</span>"
        "<span>RECOMMENDED ANALYST ACTION</span></div>",
        unsafe_allow_html=True,
    )

    recs = recommendations_to_list(get(row, "defense_recommendations"))
    if recs:
        html = ""
        for r in recs[:6]:
            html += f"""<div class="po-rec-item"><span class="po-rec-arrow">→</span><span>{r}</span></div>"""
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='po-field-muted'>No defense recommendation generated for this row.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# RENDER: EVENT STREAM (Section 10)
# ============================================================================

def render_event_stream(row: pd.Series) -> None:
    st.markdown('<div class="po-panel">', unsafe_allow_html=True)
    st.markdown(
        "<div class='po-panel-title'><span>EVENT STREAM</span>"
        "<span>DERIVED FROM SELECTED ROW</span></div>",
        unsafe_allow_html=True,
    )

    ts_label = fmt_ts(get(row, "current_timestamp"))
    events = []

    sig_count = get(row, "significant_feature_count")
    if sig_count is not None and int(sig_count) > 0:
        events.append("BEHAVIORAL CHANGE DETECTED")

    for r in evidence_to_rows(get(row, "behavior_evidence") or get(row, "evidence"))[:4]:
        arrow = "↑" if (r["pct"] or 0) > 0 else ("↓" if (r["pct"] or 0) < 0 else "→")
        events.append(f"{r['feature'].replace('_', ' ').upper()} {arrow} {fmt_pct(r['pct'])}")

    if recommendations_to_list(get(row, "defense_recommendations")):
        events.append("ANALYST ACTION GENERATED")

    if not events:
        events.append("NO EVENT DETAIL AVAILABLE FOR THIS ROW")

    html = "".join(
        f"""<div class="po-event"><div class="po-event-ts">{ts_label.split(' ')[-1] if ' ' in ts_label else ts_label}</div>
        <div class="po-event-txt">{e}</div></div>"""
        for e in events
    )
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# RENDER: RAW PREDICTION STREAM (Section 9)
# ============================================================================

def render_data_table(df: pd.DataFrame) -> None:
    st.markdown("<div class='po-section-label'>RAW PREDICTION STREAM</div>", unsafe_allow_html=True)

    default_cols = [c for c in [
        "current_timestamp", "behavior_score", "significant_feature_count",
        "stage", "mitre_tactic", "defense_recommendations",
    ] if has(df, c)]

    show_all = st.checkbox("SHOW RAW FIELDS", value=False)
    cols_to_show = list(df.columns) if show_all else (default_cols or list(df.columns))

    st.dataframe(
        df[cols_to_show],
        width="stretch",
        height=280,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    configure_page()
    inject_css()

    dataset_name = st.sidebar.selectbox(
        "DATASET",
        list(DATASETS.keys()),
        key="dataset_selector",
    )

    data_path = DATASETS[dataset_name]

    df = load_data(str(data_path))

    render_header()

    if df is None:
        render_error_panel(
            f"<code>{data_path.name}</code> was not found in the working directory."
        )
        return
    if df.empty:
        render_error_panel("The prediction file was found but contains no rows.")
        return

    filtered_df, selected_row = render_sidebar(df)

    render_kpi_strip(selected_row, len(filtered_df))

    if selected_row is None:
        st.info("Adjust filters in the left panel to select a row for review.")
        return

    col_left, col_right = st.columns([1.55, 1])
    with col_left:
        render_forecast_panel(selected_row)
        render_behavior_panel(selected_row)
    with col_right:
        render_threat_context(selected_row)
        render_defense_panel(selected_row)
        render_event_stream(selected_row)

    render_timeline(filtered_df, selected_row)
    render_data_table(filtered_df)


if __name__ == "__main__":
    main()
