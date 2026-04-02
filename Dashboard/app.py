from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
METRICS_PATH = ROOT_DIR / "Models" / "model-metrics.json"


@st.cache_data
def load_monitor_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    department_summary = pd.read_csv(DATA_DIR / "department-risk-summary.csv", parse_dates=["date"])
    system_summary = pd.read_csv(DATA_DIR / "system-risk-summary.csv", parse_dates=["date"])
    risk_trends = pd.read_csv(DATA_DIR / "risk-trends.csv", parse_dates=["date"])
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return department_summary, system_summary, risk_trends, metrics


def main() -> None:
    st.set_page_config(page_title="Campus Cyber Risk Monitor", layout="wide")
    st.title("AI Powered Campus Cybersecurity Risk Monitor")
    st.caption(
        "Predictive monitoring for higher education networks using synthetic event logs, machine learning, "
        "anomaly detection, and interpretable risk scoring."
    )

    if not (DATA_DIR / "department-risk-summary.csv").exists():
        st.warning("Run `python risk-scoring.py` first so the dashboard has data to display.")
        return

    department_summary, system_summary, risk_trends, metrics = load_monitor_outputs()
    available_departments = ["All departments"] + sorted(department_summary["department"].unique().tolist())
    selected_department = st.sidebar.selectbox("Department view", available_departments)

    latest_date = pd.to_datetime(metrics["latest_snapshot_date"])
    high_risk_departments = int(metrics["high_risk_departments_latest"])
    high_risk_systems = int(metrics["high_risk_systems_latest"])
    avg_risk_score = round(float(department_summary["avg_risk_score"].mean()), 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest snapshot", latest_date.strftime("%Y-%m-%d"))
    col2.metric("Average department risk", avg_risk_score)
    col3.metric("High-risk departments", high_risk_departments)
    col4.metric("High-risk systems", high_risk_systems)

    benchmark_col, question_col = st.columns([1.1, 1.3])
    with benchmark_col:
        st.subheader("Model Benchmarks")
        benchmark_df = pd.DataFrame(
            [
                {"Approach": "Predictive ML", **metrics["ml_model"]},
                {"Approach": "Rule based baseline", **metrics["rule_based_baseline"]},
                {"Approach": "Anomaly detector", **metrics["anomaly_detector"]},
            ]
        )
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    with question_col:
        st.subheader("Research Framing")
        st.markdown(
            "- Can tracking how a user normally acts help us guess when they might get hacked?\n"
            "- Which specific internet or computer activities are the biggest warning signs of danger?\n"
            "- What is the difference between spotting weird behavior versus just looking for broken rules?"
        )
        feature_df = pd.DataFrame(metrics["top_features"])
        st.bar_chart(feature_df.set_index("feature"))

    department_view = department_summary.copy()
    system_view = system_summary.copy()
    trend_view = risk_trends.copy()

    if selected_department != "All departments":
        department_view = department_view.loc[department_view["department"] == selected_department]
        system_view = system_view.loc[system_view["department"] == selected_department]
        trend_view = trend_view.loc[trend_view["department"] == selected_department]

    st.subheader("Department Snapshot")
    st.dataframe(
        department_view.sort_values("avg_risk_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("System Snapshot")
    st.dataframe(
        system_view.sort_values("final_risk_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Risk Trend Over Time")
    if selected_department == "All departments":
        line_df = (
            trend_view.groupby(["date", "department"], as_index=False)["final_risk_score"]
            .mean()
            .pivot(index="date", columns="department", values="final_risk_score")
            .sort_index()
        )
    else:
        line_df = (
            trend_view.groupby(["date", "system"], as_index=False)["final_risk_score"]
            .mean()
            .pivot(index="date", columns="system", values="final_risk_score")
            .sort_index()
        )
    st.line_chart(line_df)

    st.subheader("Latest Signal Breakdown")
    signal_columns = [
        "login_failures",
        "suspicious_ip_accesses",
        "unusual_file_downloads",
        "new_device_connections",
        "phishing email signals",
        "privileged_access_attempts",
    ]
    signal_df = (
        system_view[["system"] + signal_columns]
        .set_index("system")
        .sort_values("suspicious_ip_accesses", ascending=False)
    )
    st.bar_chart(signal_df)


if __name__ == "__main__":
    main()
