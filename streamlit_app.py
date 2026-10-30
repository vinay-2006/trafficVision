"""
TrafficVision — AI-Powered Intelligent Traffic Monitoring & Analytics Dashboard
Built with Streamlit and Plotly.
"""
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import TrafficPipeline
from src.perspective import PerspectiveTransformer, example_calibration
from src.reporting import ReportGenerator

# Page Configuration
st.set_page_config(
    page_title="TrafficVision — AI Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge-critical { background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-high { background-color: #f97316; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-medium { background-color: #eab308; color: black; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-low { background-color: #22c55e; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-calib { background-color: #0ea5e9; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
    .badge-uncalib { background-color: #64748b; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "annotated_video_path" not in st.session_state:
    st.session_state.annotated_video_path = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "video_path_to_process" not in st.session_state:
    st.session_state.video_path_to_process = None


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.title("TrafficVision")
    st.caption("AI-Powered Traffic Monitoring & Analytics")
    st.divider()

    st.subheader("⚙️ Pipeline Configuration")
    conf_thresh = st.slider("Detection Confidence", 0.10, 0.90, 0.35, 0.05)
    model_weights = st.text_input("YOLO Model Weights", value="yolo11n.pt")
    frame_skip = st.select_slider("Frame Skip (CPU Speedup)", options=[1, 2, 3], value=1,
                                  help="1 = process every frame, 2 = every 2nd frame (2x faster on CPU)")

    st.subheader("📐 Calibration & Geometry")
    use_calib = st.checkbox("Enable Perspective Calibration", value=True)
    calib_file = st.file_uploader("Upload calibration.json", type=["json"])

    default_sample_calib = "data/sample_calibration.json"
    active_calib_path = None
    if calib_file is not None:
        t_calib = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        t_calib.write(calib_file.read())
        t_calib.close()
        active_calib_path = t_calib.name
        st.success("Custom calibration loaded.")
    elif use_calib and Path(default_sample_calib).exists():
        active_calib_path = default_sample_calib
        st.warning("⚠️ Using illustrative sample calibration (unverified physical scale; speeds are approximate and not certified ground truth).")

    st.subheader("🚨 Incident Thresholds")
    stat_sec = st.slider("Stationary Vehicle Timeout (s)", 1.5, 8.0, 3.0, 0.5)
    free_flow_spd = st.slider("Free-Flow Reference Speed (km/h)", 30, 120, 60, 5)

    st.divider()
    st.caption("v1.0.0 | Engine: YOLO11 + ByteTrack + Homography")


# Header Section
st.title("🚦 TrafficVision — Intelligent Monitoring Dashboard")
st.markdown("Automated vehicular traffic tracking, line-crossing counting, speed homography, congestion classification, and incident detection.")


# Video Source Selection
col_src1, col_src2 = st.columns([2, 1])
with col_src1:
    uploaded_video = st.file_uploader("Upload Traffic Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])
with col_src2:
    st.markdown("**Quick Test Sample**")
    sample_available = Path("data/sample_traffic.mp4").exists()
    use_sample = st.button("Load Built-in Sample Video", disabled=not sample_available)

if uploaded_video is not None:
    t_vid = tempfile.NamedTemporaryFile(suffix=Path(uploaded_video.name).suffix, delete=False)
    t_vid.write(uploaded_video.read())
    t_vid.close()
    st.session_state.video_path_to_process = t_vid.name
elif use_sample and sample_available:
    st.session_state.video_path_to_process = "data/sample_traffic.mp4"

video_path_to_process = st.session_state.video_path_to_process
if video_path_to_process and Path(video_path_to_process).exists():
    st.info(f"Loaded `{Path(video_path_to_process).name}` for analysis.")


# Processing Trigger
if video_path_to_process:
    st.divider()
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        start_processing = st.button("🚀 Run Full Traffic Analysis", type="primary", use_container_width=True)

    if start_processing:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(msg)

        out_video_path = str(Path("outputs") / f"annotated_{Path(video_path_to_process).stem}.mp4")
        Path("outputs").mkdir(parents=True, exist_ok=True)

        try:
            status_text.text("Initializing YOLO and ByteTrack models...")
            pipeline = TrafficPipeline(
                weights=model_weights,
                conf=conf_thresh,
                calibration_path=active_calib_path if use_calib else None,
                frame_skip=frame_skip,
                stationary_time_sec=stat_sec,
                free_flow_speed_kmh=free_flow_spd,
            )

            result = pipeline.process_video(
                source_path=video_path_to_process,
                output_path=out_video_path,
                progress_callback=update_progress,
                generate_annotated_video=True,
            )

            st.session_state.pipeline_result = result
            st.session_state.annotated_video_path = out_video_path
            st.session_state.processing_done = True
            progress_bar.progress(1.0)
            status_text.success(f"Processing complete! ({result['frames_processed']} frames in {result['processing_duration_sec']:.1f}s @ {result['processing_fps']:.1f} FPS)")
        except Exception as e:
            status_text.error(f"Processing failed: {str(e)}")
            st.exception(e)


# Display Analytics Dashboard when results are available
if st.session_state.processing_done and st.session_state.pipeline_result:
    res = st.session_state.pipeline_result
    meta = res["video_metadata"]
    counts = res["vehicle_counts"]
    spd = res["speed_statistics"]
    cong = res["congestion_statistics"]
    inc_summary = res["incident_summary"]
    incidents = res["incidents"]

    tabs = st.tabs([
        "📊 Overview",
        "🎥 Video Monitoring",
        "📈 Traffic Analytics",
        "⚠️ Incident Monitoring",
        "📥 Reports & Export",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Key Performance Indicators")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Unique Vehicles Tracked</div>
                <div class="metric-value">{counts['unique_tracked']}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Line Crossings</div>
                <div class="metric-value">{counts['total_crossings']}</div>
                <small style="color: #94a3b8;">In: {counts['vehicles_in']} | Out: {counts['vehicles_out']}</small>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            calib_status = spd.get('calibration_status', 'illustrative')
            # Map formal 4-state CalibrationStatus to human-readable dashboard badge
            _CALIB_BADGE_MAP = {
                'user_attested': '<span class="badge-calib">USER-ATTESTED</span>',
                'verified':      '<span class="badge-calib">VERIFIED</span>',
                'illustrative':  '<span class="badge-uncalib">ILLUSTRATIVE SAMPLE</span>',
                'uncalibrated':  '<span class="badge-uncalib">UNCALIBRATED</span>',
                'calibrated':    '<span class="badge-calib">USER-ATTESTED</span>',  # legacy
            }
            calib_badge = _CALIB_BADGE_MAP.get(calib_status, '<span class="badge-uncalib">UNCALIBRATED</span>')

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Speed ({calib_badge})</div>
                <div class="metric-value">{spd['avg_speed_kmh']:.1f} <span style="font-size: 1rem;">km/h</span></div>
                <small style="color: #94a3b8;">Peak: {spd['max_speed_kmh']:.1f} km/h | Outliers Excluded: {spd.get('outliers_excluded_count', 0)}</small>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            cong_lvl = cong["dominant_level"]
            cong_class = {
                "LOW": "badge-low",
                "MODERATE": "badge-medium",
                "HIGH": "badge-high",
                "SEVERE": "badge-critical"
            }.get(cong_lvl, "badge-low")
            roi_src = cong.get("roi_area_source", "nominal")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Congestion Level ({roi_src})</div>
                <div class="metric-value"><span class="{cong_class}">{cong_lvl}</span></div>
                <small style="color: #94a3b8;">Peak ROI: {cong['peak_vehicles_in_roi']} veh | Area: {cong.get('roi_area_sq_meters', 400.0):.0f} m²</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        with kpi5:
            st.metric("Total Detections Logged", counts["total_detected_instances"])
        with kpi6:
            st.metric("Potential Incident Alerts", inc_summary["total_incidents"])
        with kpi7:
            st.metric("Processing Throughput", f"{res['processing_fps']:.1f} FPS")
        with kpi8:
            st.metric("Video Duration", f"{meta['duration_sec']:.1f}s ({res['frames_processed']} frames)")

        if spd.get("speed_warning"):
            st.warning(f"⚠️ **Speed Calibration Notice**: {spd['speed_warning']}")
        elif not spd["is_calibrated"]:
            st.warning("⚠️ Speed values are derived from approximate pixel scaling. Provide a real 4-point measurement via `tools/calibrate.py` with valid provenance to obtain user-attested speed estimates.")

        # Quick breakdown charts
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.subheader("Vehicle Class Breakdown")
            class_dist = counts.get("class_distribution", {})
            if class_dist and sum(class_dist.values()) > 0:
                fig_cls = px.pie(
                    names=list(class_dist.keys()),
                    values=list(class_dist.values()),
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_cls.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
                st.plotly_chart(fig_cls, use_container_width=True)
            else:
                st.info("No vehicles crossed the counting line yet.")

        with ch_col2:
            st.subheader("Congestion Level Distribution (%)")
            dist = cong.get("congestion_level_distribution", {})
            fig_cong = px.bar(
                x=list(dist.keys()),
                y=list(dist.values()),
                labels={"x": "Level", "y": "Share (%)"},
                color=list(dist.keys()),
                color_discrete_map={"LOW": "#22c55e", "MODERATE": "#eab308", "HIGH": "#f97316", "SEVERE": "#ef4444"},
            )
            fig_cong.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300, showlegend=False)
            st.plotly_chart(fig_cong, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: VIDEO MONITORING
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Annotated Video Playback")
        vid_path = st.session_state.annotated_video_path

        if vid_path and Path(vid_path).exists():
            v_col1, v_col2 = st.columns([3, 1])
            with v_col1:
                st.video(vid_path)
            with v_col2:
                st.markdown("**Overlays Rendered**")
                st.markdown("""
                - 🟩 **Vehicle Bounding Boxes & Class**
                - 🆔 **Persistent ByteTrack IDs**
                - 🚀 **Calibrated / Estimated Speed**
                - 〰️ **Motion Trajectory Trails**
                - 🛑 **Virtual Counting Line & Totals**
                - 📐 **Configured Road ROI Polygon**
                - 🏷️ **Live Congestion Status Badge**
                - 🚨 **Incident Alert Banners**
                """)
                with open(vid_path, "rb") as vf:
                    st.download_button(
                        label="📥 Download Annotated MP4",
                        data=vf.read(),
                        file_name=f"annotated_{meta['filename']}",
                        mime="video/mp4",
                        use_container_width=True,
                    )
        else:
            st.info("No annotated video available.")

    # -------------------------------------------------------------------------
    # TAB 3: TRAFFIC ANALYTICS
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Time-Series Traffic Intelligence")
        timeline = res.get("congestion_timeline", [])
        if timeline:
            df_time = pd.DataFrame(timeline)

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                fig_active = px.line(
                    df_time, x="timestamp_sec", y=["active_vehicles", "vehicles_in_roi"],
                    labels={"value": "Count", "timestamp_sec": "Time (seconds)", "variable": "Metric"},
                    title="Active Vehicles & Road ROI Density",
                    color_discrete_sequence=["#38bdf8", "#f43f5e"],
                )
                fig_active.update_layout(height=320)
                st.plotly_chart(fig_active, use_container_width=True)

            with t_col2:
                fig_spd = px.line(
                    df_time, x="timestamp_sec", y="avg_speed_kmh",
                    labels={"avg_speed_kmh": "Speed (km/h)", "timestamp_sec": "Time (seconds)"},
                    title="Average Vehicle Speed Trend (km/h)",
                    color_discrete_sequence=["#10b981"],
                )
                fig_spd.update_layout(height=320)
                st.plotly_chart(fig_spd, use_container_width=True)

            t_col3, t_col4 = st.columns(2)
            with t_col3:
                fig_score = px.area(
                    df_time, x="timestamp_sec", y="congestion_score",
                    labels={"congestion_score": "Score (0-1)", "timestamp_sec": "Time (seconds)"},
                    title="Multi-Signal Congestion Index Over Time",
                    color_discrete_sequence=["#f59e0b"],
                )
                fig_score.update_layout(height=320)
                st.plotly_chart(fig_score, use_container_width=True)

            with t_col4:
                # Vehicle Speed Distribution
                per_track = res["speed_statistics"].get("per_track_max_speed", {})
                if per_track:
                    speeds_list = list(per_track.values())
                    fig_dist = px.histogram(
                        x=speeds_list, nbins=15,
                        labels={"x": "Observed Max Speed (km/h)", "y": "Vehicle Count"},
                        title="Vehicle Max Speed Distribution (km/h)",
                        color_discrete_sequence=["#6366f1"],
                    )
                    fig_dist.update_layout(height=320)
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("Insufficient track speed history for histogram.")
        else:
            st.info("No timeline data recorded.")

    # -------------------------------------------------------------------------
    # TAB 4: INCIDENT MONITORING
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Potential Traffic Incidents & Anomaly Alerts (Rule-Based)")

        st.info("ℹ️ **Human Verification Mandatory**: All logged incidents are heuristic, rule-based indicators intended to assist traffic operations center personnel. They do NOT constitute confirmed collision or legal determinations.")

        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        sev_counts = inc_summary.get("incidents_by_severity", {})
        with i_col1:
            st.metric("Critical Alerts", sev_counts.get("CRITICAL", 0))
        with i_col2:
            st.metric("High Severity", sev_counts.get("HIGH", 0))
        with i_col3:
            st.metric("Medium Severity", sev_counts.get("MEDIUM", 0))
        with i_col4:
            st.metric("Low Severity", sev_counts.get("LOW", 0))

        if incidents:
            df_inc = pd.DataFrame(incidents)
            display_cols = ["event_id", "title", "severity", "timestamp_sec", "vehicle_ids", "description", "evidence", "confidence"]
            st.dataframe(df_inc[display_cols], use_container_width=True)
        else:
            st.success("✅ No potential traffic anomalies or incidents detected during this session.")

    # -------------------------------------------------------------------------
    # TAB 5: REPORTS & EXPORT
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Data Export & Audit Reports")
        st.markdown("Download full analytical artifacts generated from the video processing run.")

        rep = ReportGenerator(res)
        rep_json = rep.to_json()
        rep_csv = rep.to_incident_csv()
        rep_metrics = rep.to_metrics_csv()

        # Build ZIP archive in temp directory
        zip_tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        zip_tmp.close()
        rep.export_zip(zip_tmp.name, include_video_path=st.session_state.annotated_video_path)
        with open(zip_tmp.name, "rb") as zf:
            zip_bytes = zf.read()
        Path(zip_tmp.name).unlink(missing_ok=True)

        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        with exp_col1:
            st.download_button(
                "📦 Download Complete ZIP Report",
                data=zip_bytes,
                file_name=f"TrafficVision_Report_{meta['filename']}.zip",
                mime="application/zip",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                "📄 Download JSON Analytics",
                data=rep_json,
                file_name=f"traffic_analytics_{meta['filename']}.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col3:
            st.download_button(
                "⚠️ Download Incident CSV",
                data=rep_csv,
                file_name=f"incidents_{meta['filename']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col4:
            st.download_button(
                "📈 Download Metrics CSV",
                data=rep_metrics,
                file_name=f"traffic_metrics_{meta['filename']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Executive Report Preview")
        st.markdown(rep.build_summary_markdown())
