"""Judge-facing local interface for the frozen PhoenixOrder inference pipeline."""

import html
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.run_offline_inference import infer_file  # noqa: E402


CHECKPOINT_PATH = PROJECT_ROOT / "multitask_delta_lstm_checkpoint.pth"
SCALER_PATH = PROJECT_ROOT / "multitask_delta_lstm_scaler.joblib"
FEATURES_PATH = PROJECT_ROOT / "multitask_delta_lstm_features.json"
OFFLINE_THRESHOLD = 0.390906572342

st.set_page_config(
    page_title="PhoenixOrder_iitrpr | Cyber Defense Intelligence",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _install_theme():
    st.markdown(
        """
        <style>
        :root {
          --canvas: #091118;
          --surface: #101a24;
          --surface-raised: #14212d;
          --border: #263744;
          --border-soft: #1b2a36;
          --text: #e6edf3;
          --muted: #95a7b5;
          --faint: #6d8190;
          --teal: #55c7b5;
          --green: #67c79e;
          --amber: #e4ae62;
          --red: #d97878;
          --blue: #86b6d8;
        }
        .stApp, [data-testid="stAppViewContainer"] {
          background: radial-gradient(ellipse at 50% -18%, #142632 0%, var(--canvas) 54%);
          color: var(--text);
        }
        [data-testid="stHeader"] { background: rgba(9, 17, 24, .82); }
        [data-testid="stMainBlockContainer"] {
          max-width: 1540px;
          padding-top: 1.35rem;
          padding-bottom: 2.5rem;
        }
        h1, h2, h3 { color: var(--text) !important; letter-spacing: -.025em; }
        p, label, li { color: #c4d0d9; }
        [data-testid="stCaptionContainer"] p { color: var(--muted); }
        [data-testid="stFileUploaderDropzone"] {
          background: #0d1821;
          border: 1px dashed #426071;
          border-radius: 9px;
        }
        [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--teal); }
        [data-testid="stBaseButton-primary"] {
          background: #197d78;
          border: 1px solid #3faaa0;
          color: #f4fffd;
          border-radius: 6px;
          font-weight: 650;
        }
        [data-testid="stBaseButton-primary"]:hover {
          background: #218d85;
          border-color: #67d3c6;
        }
        [data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {
          border-color: #3a4d5a;
          color: var(--text);
          border-radius: 6px;
        }
        [data-testid="stDataFrame"] {
          border: 1px solid var(--border);
          border-radius: 8px;
          overflow: hidden;
        }
        [data-testid="stExpander"] {
          background: rgba(16, 26, 36, .8);
          border: 1px solid var(--border-soft);
          border-radius: 8px;
        }
        [data-testid="stMetric"] {
          background: transparent;
          border: 0;
        }
        .phoenix-hero {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 2rem;
          padding: 1.45rem 1.65rem;
          background: linear-gradient(112deg, rgba(18, 34, 45, .98), rgba(13, 23, 32, .96));
          border: 1px solid #304552;
          border-radius: 10px;
          box-shadow: 0 14px 42px rgba(0, 0, 0, .18);
          margin-bottom: 1.1rem;
        }
        .phoenix-brand { color: #f3f7fa; font-size: clamp(1.8rem, 3vw, 2.55rem); font-weight: 760; letter-spacing: -.045em; line-height: 1.1; }
        .phoenix-tagline { color: #c7d5de; font-size: 1.03rem; margin-top: .5rem; letter-spacing: .01em; }
        .phoenix-event { color: #7f9aaa; font: 600 .72rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .12em; margin-top: .6rem; text-transform: uppercase; }
        .phoenix-overline { color: #75b8b1; font: 650 .66rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .16em; margin-bottom: .65rem; }
        .phoenix-status { min-width: 225px; text-align: right; }
        .online-pill { display: inline-flex; align-items: center; gap: .5rem; padding: .42rem .68rem; border: 1px solid #315e55; border-radius: 5px; background: rgba(32, 91, 73, .2); color: #a7ddc3; font: 700 .69rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .09em; }
        .online-dot { width: 7px; height: 7px; background: var(--green); border-radius: 50%; box-shadow: 0 0 9px rgba(103, 199, 158, .5); }
        .offline-mode { color: #d7e2e8; margin-top: .75rem; font: 700 .72rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .1em; }
        .offline-note { color: #78909e; margin-top: .35rem; font-size: .75rem; }
        .section-head { display: flex; align-items: baseline; gap: .72rem; border-bottom: 1px solid var(--border-soft); padding-bottom: .6rem; margin: 1.25rem 0 .9rem; color: #dbe5eb; font-size: .83rem; font-weight: 720; letter-spacing: .105em; text-transform: uppercase; }
        .section-index { color: #5f9b98; font: 650 .69rem ui-monospace, SFMono-Regular, Menlo, monospace; }
        .section-note { color: var(--muted); font-size: .82rem; margin-top: -.45rem; margin-bottom: .8rem; }
        .kpi-card { min-height: 106px; padding: .86rem .95rem; background: linear-gradient(150deg, #13212b, #0f1922); border: 1px solid #293a46; border-top: 2px solid #456474; border-radius: 7px; }
        .kpi-card.teal { border-top-color: var(--teal); }
        .kpi-card.green { border-top-color: var(--green); }
        .kpi-card.amber { border-top-color: var(--amber); }
        .kpi-card.red { border-top-color: var(--red); }
        .kpi-label { color: #8fa3b1; font: 650 .65rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .095em; text-transform: uppercase; }
        .kpi-value { color: #edf3f6; margin-top: .58rem; font-size: 1.62rem; font-weight: 720; letter-spacing: -.035em; line-height: 1.05; }
        .kpi-foot { color: #718693; margin-top: .36rem; font-size: .7rem; }
        .soft-panel { background: linear-gradient(145deg, rgba(18, 29, 39, .98), rgba(13, 22, 30, .98)); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.05rem; }
        .panel-title { color: #dce6eb; font-size: .76rem; font-weight: 740; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .75rem; }
        .mono { color: #a9c4d0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem; }
        .legend { display: flex; flex-wrap: wrap; gap: 1rem; color: #a8b8c3; font: 650 .65rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .07em; margin: -.15rem 0 .5rem; }
        .legend-item { display: inline-flex; align-items: center; gap: .42rem; }
        .legend-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
        .legend-line { display: inline-block; width: 17px; border-top: 2px dashed #d9aa67; }
        .detail-label { color: #78909e; font: 650 .64rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .1em; text-transform: uppercase; }
        .detail-value { color: #e3ebef; font-size: 1.04rem; font-weight: 650; margin-top: .24rem; }
        .stage-unknown { padding: .75rem .85rem; background: rgba(117, 133, 146, .09); border-left: 3px solid #81909b; border-radius: 4px; color: #bdc9d0; }
        .model-chip { display: inline-block; padding: .3rem .48rem; background: #142430; border: 1px solid #2a414d; border-radius: 4px; color: #b7cbd5; font: 600 .7rem ui-monospace, SFMono-Regular, Menlo, monospace; }
        .flow-step { padding: .55rem .7rem; background: #111e28; border: 1px solid #293a46; border-radius: 5px; color: #d2dce2; font-size: .79rem; text-align: center; }
        .flow-arrow { color: #5c9e98; text-align: center; line-height: 1.1; padding: .12rem; }
        .footer { border-top: 1px solid var(--border-soft); color: #7f929f; margin-top: 2.2rem; padding: 1rem .1rem .4rem; display: flex; justify-content: space-between; gap: 1rem; font-size: .75rem; }
        .footer strong { color: #a9bec9; letter-spacing: .03em; }
        @media (max-width: 760px) {
          .phoenix-hero { align-items: flex-start; flex-direction: column; padding: 1.15rem; }
          .phoenix-status { text-align: left; }
          .footer { flex-direction: column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section(index, title, note=None):
    st.markdown(
        f'<div class="section-head"><span class="section-index">{index}</span>{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    if note:
        st.markdown(f'<div class="section-note">{html.escape(note)}</div>', unsafe_allow_html=True)


def _kpi(label, value, foot, tone="neutral"):
    return (
        f'<div class="kpi-card {tone}"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-foot">{html.escape(foot)}</div></div>'
    )


def _run_inference(uploaded_file):
    """Stage the upload locally, then delegate to the existing inference pipeline."""
    safe_name = Path(uploaded_file.name).name
    with tempfile.TemporaryDirectory(prefix="phoenixorder_demo_") as temp_dir:
        input_path = Path(temp_dir) / safe_name
        output_path = Path(temp_dir) / "predictions.csv"
        input_path.write_bytes(uploaded_file.getvalue())
        predictions = infer_file(
            input_path,
            output_path,
            CHECKPOINT_PATH,
            SCALER_PATH,
            FEATURES_PATH,
        )
        return predictions.copy()


def _probability_timeline(frame, threshold):
    ordered = frame.sort_values("timestamp").copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"])
    figure = go.Figure()

    for _, segment in ordered.groupby("segment_id", sort=True):
        figure.add_trace(go.Scatter(
            x=segment["timestamp"],
            y=segment["attack_probability"],
            mode="lines",
            line={"color": "#607482", "width": 1.35},
            name="Probability trend",
            hoverinfo="skip",
            showlegend=False,
        ))

    for predicted_class, color in (("attack", "#cf7773"), ("benign", "#69b99a")):
        points = ordered[ordered["predicted_class"] == predicted_class]
        figure.add_trace(go.Scatter(
            x=points["timestamp"],
            y=points["attack_probability"],
            mode="markers",
            name=predicted_class.upper(),
            marker={"color": color, "size": 7, "line": {"color": "#0c151d", "width": 1}},
            customdata=points[["segment_id", "attack_stage", "behavioral_change_score"]],
            hovertemplate=(
                "%{x}<br>Attack probability: %{y:.4f}<br>"
                "Segment: %{customdata[0]}<br>Stage: %{customdata[1]}<br>"
                "Behavioral change score: %{customdata[2]:.4f}<extra></extra>"
            ),
        ))

    figure.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#dfad65",
        line_width=1.4,
    )
    figure.update_layout(
        height=440,
        margin={"l": 16, "r": 22, "t": 18, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0e1821",
        font={"family": "Inter, system-ui, sans-serif", "color": "#b9c7d0", "size": 11},
        xaxis={"title": "Timestamp", "gridcolor": "#22313c", "linecolor": "#344652", "zeroline": False},
        yaxis={"title": "Multitask attack probability", "range": [0, 1], "gridcolor": "#22313c", "linecolor": "#344652", "zeroline": False},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 10}},
        hovermode="closest",
    )
    return figure


def _read_json(value, default):
    try:
        result = json.loads(value)
        return result if result is not None else default
    except (TypeError, json.JSONDecodeError):
        return default


def _render_detail(row):
    _section("05", "Behavioral intelligence", "Selected window assessment from model outputs")
    threat_columns = st.columns(4)
    details = [
        ("Attack probability", f"{float(row['attack_probability']):.4f}"),
        ("Behavioral change", f"{float(row['behavioral_change_score']):.4f}"),
        ("Prediction", str(row["predicted_class"]).upper()),
        ("Attack stage", str(row["attack_stage"]).upper()),
    ]
    for column, (label, value) in zip(threat_columns, details):
        column.markdown(
            f'<div class="soft-panel"><div class="detail-label">{html.escape(label)}</div>'
            f'<div class="detail-value">{html.escape(value)}</div></div>',
            unsafe_allow_html=True,
        )

    stage = str(row["attack_stage"])
    if stage == "Unknown":
        st.markdown(
            '<div class="stage-unknown"><strong>UNKNOWN</strong><br>'
            "Insufficient behavioral evidence for confident stage classification.</div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="panel-title">Behavioral evidence</div>', unsafe_allow_html=True)
        evidence = _read_json(row["behavioral_change_evidence"], [])
        if evidence:
            for item in evidence:
                st.markdown(f"- {item}")
        else:
            st.caption("No significant behavioral evidence was produced for this window.")

        changes = _read_json(row["significant_feature_changes"], [])
        st.markdown('<div class="panel-title">Significant feature changes</div>', unsafe_allow_html=True)
        if changes:
            change_frame = pd.DataFrame(changes)
            preferred = [
                "feature", "direction", "current_value", "predicted_value",
                "normalized_change", "percentage_change", "magnitude",
            ]
            shown = [column for column in preferred if column in change_frame.columns]
            st.dataframe(change_frame[shown], width="stretch", hide_index=True, height=220)
        else:
            st.caption("No feature change crossed the existing significance cutoff.")

    with right:
        _section("06", "Attack stage / MITRE ATT&CK")
        tactic = str(row["mitre_tactic"] or "Unknown")
        tactic_id = str(row["mitre_tactic_id"] or "—")
        st.markdown(
            f'<div class="soft-panel"><div class="detail-label">Tactic</div>'
            f'<div class="detail-value">{html.escape(tactic)}</div>'
            f'<div class="mono">{html.escape(tactic_id)}</div></div>',
            unsafe_allow_html=True,
        )
        techniques = _read_json(row["mitre_techniques"], [])
        if techniques:
            st.markdown("**Mapped techniques**")
            for technique in techniques:
                technique_id = html.escape(str(technique.get("id", "—")))
                technique_name = html.escape(str(technique.get("name", "Unknown")))
                st.markdown(
                    f'<div class="soft-panel" style="margin:.35rem 0">'
                    f'<span class="model-chip">{technique_id}</span> '
                    f'{technique_name}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No MITRE technique mapped for this stage.")

        _section("07", "Defense recommendations")
        recommendations = _read_json(row["defense_recommendations"], [])
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")


def _show_explanation():
    with st.expander("How PhoenixOrder Works", expanded=False):
        steps = [
            "Raw Network Traffic", "Feature Extraction", "5-Minute Network States",
            "12-State Temporal Context", "Multitask Delta-LSTM",
            "Attack Probability + Next-State Prediction", "Behavioral Analysis",
            "MITRE ATT&CK Mapping", "Defense Recommendations",
        ]
        for index, step in enumerate(steps):
            st.markdown(f'<div class="flow-step">{html.escape(step)}</div>', unsafe_allow_html=True)
            if index < len(steps) - 1:
                st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)


def _show_model_card(threshold):
    _section("08", "Model / system information")
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f"""
            <div class="soft-panel">
              <div class="panel-title">Inference configuration</div>
              <div class="detail-label">Model</div><div class="detail-value">Multitask Delta-LSTM</div><br>
              <div class="detail-label">Input</div><div class="mono">32 temporal network features</div><br>
              <div class="detail-label">Context</div><div class="mono">12 network states</div><br>
              <div class="detail-label">Mode</div><div class="mono">Offline inference</div><br>
              <div class="detail-label">Threshold</div><div class="mono">{threshold:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="soft-panel">
              <div class="panel-title">Benchmark results · fixed test split</div>
              <div class="detail-value">Multitask Delta-LSTM</div>
              <div class="mono">F1 0.4675 &nbsp;·&nbsp; Recall 1.0000 &nbsp;·&nbsp; FPR 0.7500</div>
              <hr style="border-color:#263744; margin:.85rem 0">
              <div class="detail-value">Logistic Regression baseline</div>
              <div class="mono">F1 0.4045</div>
              <p style="color:#8295a1;font-size:.72rem;margin:.75rem 0 0">
                Benchmark metrics are fixed evaluation results, not live performance on the uploaded file.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _display_app():
    _install_theme()
    st.markdown(
        """
        <div class="phoenix-hero">
          <div>
            <div class="phoenix-overline">SIH 2026 &nbsp; / &nbsp; CYBER DEFENSE INTELLIGENCE</div>
            <div class="phoenix-brand">PhoenixOrder_iitrpr</div>
            <div class="phoenix-tagline">AI-Powered Proactive Cyber Defense</div>
            <div class="phoenix-event">SIH 2026 &nbsp;•&nbsp; SIH26153</div>
          </div>
          <div class="phoenix-status">
            <div class="online-pill"><span class="online-dot"></span>SYSTEM ONLINE</div>
            <div class="offline-mode">OFFLINE INFERENCE MODE</div>
            <div class="offline-note">Local processing · No cloud APIs</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-head"><span class="section-index">00</span>COMMAND CENTER</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload CSE-CIC-IDS2018-style flow data to create a local, time-aligned network assessment."
    )

    missing = [path.name for path in (CHECKPOINT_PATH, SCALER_PATH, FEATURES_PATH) if not path.is_file()]
    if missing:
        st.error("Saved model artifacts are unavailable: " + ", ".join(missing))
        st.stop()

    upload_column, action_column = st.columns([4, 1])
    with upload_column:
        uploaded_file = st.file_uploader(
            "Network flow CSV",
            type=["csv"],
            help="Uploaded data is staged in a local temporary directory and processed by the existing offline pipeline.",
            label_visibility="collapsed",
        )
    with action_column:
        st.markdown("<div style='height:1.82rem'></div>", unsafe_allow_html=True)
        run_clicked = st.button(
            "Run local analysis",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

    if run_clicked and uploaded_file is not None:
        try:
            with st.spinner("Aggregating local network states and running frozen inference…"):
                result = _run_inference(uploaded_file)
            st.session_state["phoenix_predictions"] = result
            st.session_state["phoenix_upload_name"] = Path(uploaded_file.name).name
        except Exception as error:
            st.error(f"Inference failed: {error}")
            st.stop()

    predictions = st.session_state.get("phoenix_predictions")
    if predictions is None:
        _section("01", "System overview")
        st.info("Upload a CSV and run local analysis to populate the command center.")
        _show_explanation()
        _show_model_card(OFFLINE_THRESHOLD)
        st.markdown(
            '<div class="footer"><strong>PhoenixOrder_iitrpr</strong>'
            '<span>SIH 2026 • SIH26153</span><span>Offline AI Cyber Defense Prototype</span></div>',
            unsafe_allow_html=True,
        )
        return

    threshold = float(predictions["classification_threshold"].iloc[0])
    attack_count = int((predictions["predicted_class"] == "attack").sum())
    benign_count = int((predictions["predicted_class"] == "benign").sum())
    highest_probability = float(predictions["attack_probability"].max())
    average_probability = float(predictions["attack_probability"].mean())
    uploaded_name = html.escape(st.session_state.get("phoenix_upload_name", "uploaded flow file"))

    _section("01", "System overview", f"Active input: {uploaded_name} · inference threshold loaded from the saved checkpoint")
    kpi_columns = st.columns(5)
    cards = [
        ("Temporal windows", f"{len(predictions):,}", "5-minute states with valid context", "teal"),
        ("Flagged windows", f"{attack_count:,}", f"{benign_count:,} predicted benign", "amber" if attack_count else "green"),
        ("Highest probability", f"{highest_probability:.3f}", "Classifier output maximum", "red" if highest_probability >= .85 else "amber" if highest_probability >= threshold else "green"),
        ("Average probability", f"{average_probability:.3f}", "Across current input windows", "neutral"),
        ("Current threshold", f"{threshold:.6f}", "Validation-selected · unchanged", "teal"),
    ]
    for column, card in zip(kpi_columns, cards):
        column.markdown(_kpi(*card), unsafe_allow_html=True)

    _section("02", "Threat / infiltration timeline", "Classifier probability by prediction timestamp; lines do not bridge timestamp-gap segments")
    st.markdown(
        '<div class="legend">'
        '<span class="legend-item"><span class="legend-dot" style="background:#cf7773"></span>ATTACK</span>'
        '<span class="legend-item"><span class="legend-dot" style="background:#69b99a"></span>BENIGN</span>'
        '<span class="legend-item"><span class="legend-line"></span>THRESHOLD</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_probability_timeline(predictions, threshold), width="stretch")

    _section("03", "Flagged network activity", "Select a row to open its intelligence detail")
    filters = st.columns([1.1, 1.3, 1])
    selected_classes = filters[0].multiselect(
        "Predicted class", ["attack", "benign"], default=["attack"], key="class_filter"
    )
    stages = sorted(predictions["attack_stage"].dropna().astype(str).unique())
    selected_stages = filters[1].multiselect(
        "Attack stage", stages, default=stages, key="stage_filter"
    )
    segments = sorted(predictions["segment_id"].astype(int).unique().tolist())
    selected_segments = filters[2].multiselect(
        "Segment", segments, default=segments, key="segment_filter"
    )

    filtered = predictions[
        predictions["predicted_class"].isin(selected_classes)
        & predictions["attack_stage"].astype(str).isin(selected_stages)
        & predictions["segment_id"].astype(int).isin(selected_segments)
    ].copy().sort_values("timestamp", ascending=False).reset_index(drop=True)
    st.caption(f"{len(filtered):,} windows match the current filters · click a row for details")
    table_columns = [
        "timestamp", "attack_probability", "behavioral_change_score",
        "attack_stage", "segment_id",
    ]
    if filtered.empty:
        st.info("No windows match the selected filters.")
        selection_event = None
    else:
        selection_event = st.dataframe(
            filtered[table_columns],
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "attack_probability": st.column_config.NumberColumn("Attack probability", format="%.4f"),
                "behavioral_change_score": st.column_config.NumberColumn("Behavioral change", format="%.4f"),
                "attack_stage": st.column_config.TextColumn("Attack stage"),
                "segment_id": st.column_config.NumberColumn("Segment", format="%d"),
            },
            width="stretch",
            hide_index=True,
            height=310,
            on_select="rerun",
            selection_mode="single-row",
            key="activity_table",
        )

    if selection_event is not None:
        selected_rows = selection_event.selection.rows
        if selected_rows:
            _render_detail(filtered.iloc[selected_rows[0]])
        else:
            st.caption("Choose a table row to inspect behavioral evidence, ATT&CK mapping, and the existing response guidance.")

    st.download_button(
        "Export complete inference results",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(st.session_state.get('phoenix_upload_name', 'network')).stem}_predictions.csv",
        mime="text/csv",
    )

    _show_explanation()
    _show_model_card(threshold)
    st.markdown(
        '<div class="footer"><strong>PhoenixOrder_iitrpr</strong>'
        '<span>SIH 2026 • SIH26153</span><span>Offline AI Cyber Defense Prototype</span></div>',
        unsafe_allow_html=True,
    )


_display_app()
