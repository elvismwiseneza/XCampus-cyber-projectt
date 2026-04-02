from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "Models"


@dataclass(frozen=True)
class PipelinePaths:
    raw_logs: Path = DATA_DIR / "raw-logs.csv"
    cleaned_logs: Path = DATA_DIR / "cleaned-logs.csv"
    risk_trends: Path = DATA_DIR / "risk-trends.csv"
    department_summary: Path = DATA_DIR / "department-risk-summary.csv"
    system_summary: Path = DATA_DIR / "system-risk-summary.csv"
    model_bundle: Path = MODELS_DIR / "anomaly-model.pkl"
    model_metrics: Path = MODELS_DIR / "model-metrics.json"


PIPELINE_PATHS = PipelinePaths()

DEPARTMENT_PROFILES = {
    "Admissions": {
        "systems": ["admissions-portal", "application-db"],
        "login_failures": 6,
        "suspicious_ip_accesses": 2,
        "unusual_file_downloads": 1,
        "new_device_connections": 2,
        "phishing_email_signals": 3,
        "privileged_access_attempts": 1,
        "risk_bias": 0.15,
    },
    "Finance": {
        "systems": ["finance-erp", "payroll-db"],
        "login_failures": 7,
        "suspicious_ip_accesses": 3,
        "unusual_file_downloads": 2,
        "new_device_connections": 2,
        "phishing_email_signals": 2,
        "privileged_access_attempts": 2,
        "risk_bias": 0.30,
    },
    "Human Resources": {
        "systems": ["hr-portal", "benefits-db"],
        "login_failures": 5,
        "suspicious_ip_accesses": 2,
        "unusual_file_downloads": 1,
        "new_device_connections": 1,
        "phishing_email_signals": 3,
        "privileged_access_attempts": 1,
        "risk_bias": 0.18,
    },
    "Library": {
        "systems": ["library-catalog", "digital-archives"],
        "login_failures": 4,
        "suspicious_ip_accesses": 1,
        "unusual_file_downloads": 2,
        "new_device_connections": 3,
        "phishing_email_signals": 1,
        "privileged_access_attempts": 1,
        "risk_bias": 0.08,
    },
    "Registrar": {
        "systems": ["student-records", "registration-api"],
        "login_failures": 6,
        "suspicious_ip_accesses": 2,
        "unusual_file_downloads": 1,
        "new_device_connections": 2,
        "phishing_email_signals": 2,
        "privileged_access_attempts": 2,
        "risk_bias": 0.20,
    },
    "Research Lab": {
        "systems": ["hpc-cluster", "research-data-vault"],
        "login_failures": 8,
        "suspicious_ip_accesses": 3,
        "unusual_file_downloads": 4,
        "new_device_connections": 3,
        "phishing_email_signals": 2,
        "privileged_access_attempts": 3,
        "risk_bias": 0.40,
    },
    "Student Services": {
        "systems": ["student-portal", "housing-api"],
        "login_failures": 5,
        "suspicious_ip_accesses": 2,
        "unusual_file_downloads": 1,
        "new_device_connections": 3,
        "phishing_email_signals": 2,
        "privileged_access_attempts": 1,
        "risk_bias": 0.12,
    },
}

EVENT_SEVERITY = {
    "login_failure": ("low", "medium"),
    "suspicious_ip_access": ("medium", "high"),
    "unusual_file_download": ("medium", "high"),
    "new_device_connection": ("low", "medium"),
    "phishing_email_signal": ("medium", "high"),
    "privileged_access_attempt": ("high", "critical"),
}

EVENT_COLUMN_MAP = {
    "login_failure": "login_failures",
    "suspicious_ip_access": "suspicious_ip_accesses",
    "unusual_file_download": "unusual_file_downloads",
    "new_device_connection": "new_device_connections",
    "phishing_email_signal": "phishing_email_signals",
    "privileged_access_attempt": "privileged_access_attempts",
}

CORE_SIGNAL_COLUMNS = list(EVENT_COLUMN_MAP.values())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sigmoid(values: np.ndarray | pd.Series | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values)))


def choose_severity(event_type: str, risk_pressure: float, rng: np.random.Generator) -> str:
    base, elevated = EVENT_SEVERITY[event_type]
    elevated_probability = min(0.82, 0.25 + (risk_pressure * 0.18))
    if rng.random() < elevated_probability:
        return elevated
    return base


def generate_ip(ip_type: str, rng: np.random.Generator) -> str:
    if ip_type == "internal":
        return f"10.{rng.integers(1, 240)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
    return f"{rng.integers(20, 220)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"


def make_event_rows(
    *,
    event_type: str,
    count: int,
    event_date: datetime,
    department: str,
    systems: list[str],
    risk_pressure: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _ in range(int(count)):
        system = str(rng.choice(systems))
        hour_bias = 0.55 if risk_pressure > 1.2 else 0.20
        off_hours = rng.random() < hour_bias
        event_hour = int(rng.choice([0, 1, 2, 3, 4, 5, 21, 22, 23])) if off_hours else int(rng.integers(7, 20))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        timestamp = event_date + timedelta(hours=event_hour, minutes=minute, seconds=second)
        ip_type = "external" if event_type in {"suspicious_ip_access", "phishing_email_signal"} or rng.random() < 0.28 else "internal"
        rows.append(
            {
                "timestamp": timestamp.isoformat(timespec="seconds"),
                "department": department,
                "system": system,
                "event_type": event_type,
                "severity": choose_severity(event_type, risk_pressure, rng),
                "source_ip": generate_ip(ip_type, rng),
                "source_ip_type": ip_type,
                "user_id": f"{department[:3].upper()}-U{rng.integers(1000, 9999)}",
                "device_id": f"{department[:3].upper()}-D{rng.integers(100, 999)}",
                "download_mb": int(rng.integers(150, 5000)) if event_type == "unusual_file_download" else 0,
                "is_new_device": int(event_type == "new_device_connection"),
            }
        )
    return rows


def generate_raw_logs(output_path: Path = PIPELINE_PATHS.raw_logs, days: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
    all_rows: list[dict[str, object]] = []

    for day_offset in range(days):
        event_date = start_date + timedelta(days=day_offset)
        weekend_factor = 0.78 if event_date.weekday() >= 5 else 1.0
        semester_factor = 1.10 if event_date.month in {9, 10, 11, 2, 3, 4} else 0.92

        for department, profile in DEPARTMENT_PROFILES.items():
            risk_pressure = max(
                0.0,
                rng.normal(0.75 + profile["risk_bias"], 0.28) + (rng.uniform(1.2, 2.7) if rng.random() < 0.12 else 0.0),
            )

            event_counts = {
                "login_failure": rng.poisson(profile["login_failures"] * weekend_factor * semester_factor * (1.0 + risk_pressure * 0.85)),
                "suspicious_ip_access": rng.poisson(profile["suspicious_ip_accesses"] * weekend_factor * semester_factor * (1.0 + risk_pressure * 1.10)),
                "unusual_file_download": rng.poisson(profile["unusual_file_downloads"] * weekend_factor * semester_factor * (1.0 + risk_pressure)),
                "new_device_connection": rng.poisson(profile["new_device_connections"] * weekend_factor * semester_factor * (1.0 + risk_pressure * 0.70)),
                "phishing_email_signal": rng.poisson(profile["phishing_email_signals"] * weekend_factor * semester_factor * (1.0 + risk_pressure * 1.15)),
                "privileged_access_attempt": rng.poisson(profile["privileged_access_attempts"] * weekend_factor * semester_factor * (1.0 + risk_pressure * 1.25)),
            }

            for event_type, count in event_counts.items():
                all_rows.extend(
                    make_event_rows(
                        event_type=event_type,
                        count=count,
                        event_date=event_date,
                        department=department,
                        systems=profile["systems"],
                        risk_pressure=risk_pressure,
                        rng=rng,
                    )
                )

    raw_logs = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)
    ensure_parent(output_path)
    raw_logs.to_csv(output_path, index=False)
    return raw_logs


def _rolling_mean(series: pd.Series, window: int = 7) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _determine_primary_driver(row: pd.Series) -> str:
    driver_weights = {
        "Login failures": row["login_failures"] * 1.0,
        "Suspicious IP access": row["suspicious_ip_accesses"] * 1.7,
        "Unusual downloads": row["unusual_file_downloads"] * 1.5,
        "New device joins": row["new_device_connections"] * 1.1,
        "Phishing patterns": row["phishing_email_signals"] * 1.8,
        "Privileged access attempts": row["privileged_access_attempts"] * 1.9,
    }
    return max(driver_weights, key=driver_weights.get)


def preprocess_logs(
    raw_path: Path = PIPELINE_PATHS.raw_logs,
    output_path: Path = PIPELINE_PATHS.cleaned_logs,
    seed: int = 42,
) -> pd.DataFrame:
    raw_logs = pd.read_csv(raw_path, parse_dates=["timestamp"])
    raw_logs["date"] = raw_logs["timestamp"].dt.floor("D")
    raw_logs["hour"] = raw_logs["timestamp"].dt.hour
    raw_logs["is_off_hours"] = ((raw_logs["hour"] < 6) | (raw_logs["hour"] >= 21)).astype(int)
    raw_logs["is_high_severity"] = raw_logs["severity"].isin(["high", "critical"]).astype(int)
    raw_logs["is_external_ip"] = (raw_logs["source_ip_type"] == "external").astype(int)

    base_group = ["date", "department", "system"]
    event_counts = (
        raw_logs.pivot_table(index=base_group, columns="event_type", values="user_id", aggfunc="count", fill_value=0)
        .rename(columns=EVENT_COLUMN_MAP)
        .reset_index()
    )
    for column in CORE_SIGNAL_COLUMNS:
        if column not in event_counts.columns:
            event_counts[column] = 0

    aggregates = (
        raw_logs.groupby(base_group)
        .agg(
            total_events=("event_type", "size"),
            unique_users=("user_id", "nunique"),
            unique_devices=("device_id", "nunique"),
            external_ip_events=("is_external_ip", "sum"),
            off_hours_events=("is_off_hours", "sum"),
            high_severity_events=("is_high_severity", "sum"),
            total_download_volume_mb=("download_mb", "sum"),
        )
        .reset_index()
    )

    dataset = event_counts.merge(aggregates, on=base_group, how="left").sort_values(base_group).reset_index(drop=True)
    dataset["external_ip_ratio"] = dataset["external_ip_events"] / dataset["total_events"].clip(lower=1)
    dataset["off_hours_ratio"] = dataset["off_hours_events"] / dataset["total_events"].clip(lower=1)
    dataset["high_severity_ratio"] = dataset["high_severity_events"] / dataset["total_events"].clip(lower=1)

    rolling_group = dataset.groupby(["department", "system"], sort=False)
    for column in CORE_SIGNAL_COLUMNS + ["total_events", "external_ip_events", "off_hours_events"]:
        dataset[f"{column}_7d_avg"] = rolling_group[column].transform(_rolling_mean)
        dataset[f"{column}_spike_ratio"] = dataset[column] / (dataset[f"{column}_7d_avg"].fillna(0) + 1.0)

    profile_risk = dataset["department"].map({name: profile["risk_bias"] for name, profile in DEPARTMENT_PROFILES.items()}).fillna(0.1)
    rule_signal = (
        0.10 * dataset["login_failures"]
        + 0.24 * dataset["suspicious_ip_accesses"]
        + 0.18 * dataset["unusual_file_downloads"]
        + 0.10 * dataset["new_device_connections"]
        + 0.22 * dataset["phishing_email_signals"]
        + 0.28 * dataset["privileged_access_attempts"]
        + 1.10 * dataset["off_hours_ratio"]
        + 1.20 * dataset["external_ip_ratio"]
        + 0.20 * dataset["high_severity_ratio"]
    )
    predictive_signal = (
        rule_signal
        + 0.16 * dataset["login_failures_spike_ratio"]
        + 0.28 * dataset["suspicious_ip_accesses_spike_ratio"]
        + 0.18 * dataset["unusual_file_downloads_spike_ratio"]
        + 0.24 * dataset["phishing_email_signals_spike_ratio"]
        + 0.20 * dataset["privileged_access_attempts_spike_ratio"]
        + 0.16 * dataset["total_events_spike_ratio"]
        + 0.12 * dataset["external_ip_events_spike_ratio"]
        + 0.10 * dataset["off_hours_events_spike_ratio"]
        + profile_risk
    )

    rule_based_score = 100.0 * sigmoid(-3.6 + rule_signal)
    dataset["rule_based_score"] = rule_based_score.round(2)

    rng = np.random.default_rng(seed + 7)
    incident_probability = sigmoid(-5.3 + predictive_signal + rng.normal(0, 0.22, len(dataset)))
    dataset["incident_probability"] = incident_probability.round(4)
    dataset["incident_next_7d"] = rng.binomial(1, incident_probability)
    dataset["primary_driver"] = dataset.apply(_determine_primary_driver, axis=1)

    ensure_parent(output_path)
    dataset.to_csv(output_path, index=False)
    return dataset


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, binary_predictions: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, binary_predictions)), 4),
        "precision": round(float(precision_score(y_true, binary_predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, binary_predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, binary_predictions, zero_division=0)), 4),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, probabilities)), 4)
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def build_summary_tables(scored_dataset: pd.DataFrame, paths: PipelinePaths = PIPELINE_PATHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_date = scored_dataset["date"].max()
    latest_rows = scored_dataset.loc[scored_dataset["date"] == latest_date].copy()
    latest_rows["risk_rank"] = latest_rows["final_risk_score"].rank(method="dense", ascending=False)

    system_summary = latest_rows[
        [
            "date",
            "department",
            "system",
            "final_risk_score",
            "risk_level",
            "ml_incident_probability",
            "anomaly_score",
            "rule_based_score",
            "primary_driver",
            "login_failures",
            "suspicious_ip_accesses",
            "unusual_file_downloads",
            "new_device_connections",
            "phishing_email_signals",
            "privileged_access_attempts",
        ]
    ].sort_values("final_risk_score", ascending=False)

    department_summary = (
        latest_rows.groupby("department")
        .agg(
            monitored_systems=("system", "nunique"),
            avg_risk_score=("final_risk_score", "mean"),
            highest_risk_level=("risk_level", lambda values: "High" if "High" in values.values else ("Medium" if "Medium" in values.values else "Low")),
            systems_flagged_high=("risk_level", lambda values: int((values == "High").sum())),
            top_driver=("primary_driver", lambda values: values.mode().iat[0]),
            mean_ml_probability=("ml_incident_probability", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk_score", ascending=False)
    )
    department_summary["avg_risk_score"] = department_summary["avg_risk_score"].round(2)
    department_summary["mean_ml_probability"] = department_summary["mean_ml_probability"].round(4)
    department_summary["date"] = latest_date

    ensure_parent(paths.system_summary)
    ensure_parent(paths.department_summary)
    system_summary.to_csv(paths.system_summary, index=False)
    department_summary.to_csv(paths.department_summary, index=False)
    return department_summary, system_summary


def train_and_score(
    cleaned_path: Path = PIPELINE_PATHS.cleaned_logs,
    paths: PipelinePaths = PIPELINE_PATHS,
    seed: int = 42,
) -> dict[str, object]:
    import joblib
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    dataset = pd.read_csv(cleaned_path, parse_dates=["date"])

    categorical_features = ["department", "system"]
    numeric_features = [
        "login_failures",
        "suspicious_ip_accesses",
        "unusual_file_downloads",
        "new_device_connections",
        "phishing_email_signals",
        "privileged_access_attempts",
        "total_events",
        "unique_users",
        "unique_devices",
        "external_ip_events",
        "off_hours_events",
        "high_severity_events",
        "total_download_volume_mb",
        "external_ip_ratio",
        "off_hours_ratio",
        "high_severity_ratio",
        "login_failures_7d_avg",
        "suspicious_ip_accesses_7d_avg",
        "unusual_file_downloads_7d_avg",
        "new_device_connections_7d_avg",
        "phishing_email_signals_7d_avg",
        "privileged_access_attempts_7d_avg",
        "total_events_7d_avg",
        "login_failures_spike_ratio",
        "suspicious_ip_accesses_spike_ratio",
        "unusual_file_downloads_spike_ratio",
        "new_device_connections_spike_ratio",
        "phishing_email_signals_spike_ratio",
        "privileged_access_attempts_spike_ratio",
        "total_events_spike_ratio",
    ]

    X = dataset[categorical_features + numeric_features]
    y = dataset["incident_next_7d"]
    rule_scores = dataset["rule_based_score"]

    X_train, X_test, y_train, y_test, rule_train, rule_test = train_test_split(
        X,
        y,
        rule_scores,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    classifier_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    evaluation_model = clone(classifier_pipeline)
    evaluation_model.fit(X_train, y_train)
    ml_probabilities = evaluation_model.predict_proba(X_test)[:, 1]
    ml_predictions = (ml_probabilities >= 0.55).astype(int)
    ml_metrics = compute_metrics(y_test, ml_probabilities, ml_predictions)

    anomaly_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    train_numeric = anomaly_transformer.fit_transform(X_train[numeric_features])
    test_numeric = anomaly_transformer.transform(X_test[numeric_features])
    anomaly_model = IsolationForest(n_estimators=220, contamination=0.18, random_state=seed)
    anomaly_model.fit(train_numeric)
    anomaly_scores_raw = -anomaly_model.decision_function(test_numeric)
    anomaly_probabilities = (anomaly_scores_raw - anomaly_scores_raw.min()) / (anomaly_scores_raw.max() - anomaly_scores_raw.min() + 1e-9)
    anomaly_predictions = (anomaly_probabilities >= 0.52).astype(int)
    anomaly_metrics = compute_metrics(y_test, anomaly_probabilities, anomaly_predictions)

    rule_probabilities = rule_test.to_numpy() / 100.0
    rule_predictions = (rule_probabilities >= 0.65).astype(int)
    rule_metrics = compute_metrics(y_test, rule_probabilities, rule_predictions)

    deployment_model = clone(classifier_pipeline)
    deployment_model.fit(X, y)

    deployment_anomaly_transformer = clone(anomaly_transformer)
    full_numeric = deployment_anomaly_transformer.fit_transform(X[numeric_features])
    deployment_anomaly_model = IsolationForest(n_estimators=220, contamination=0.18, random_state=seed)
    deployment_anomaly_model.fit(full_numeric)

    dataset["ml_incident_probability"] = deployment_model.predict_proba(X)[:, 1]
    full_anomaly_scores = -deployment_anomaly_model.decision_function(full_numeric)
    dataset["anomaly_score"] = 100 * (full_anomaly_scores - full_anomaly_scores.min()) / (full_anomaly_scores.max() - full_anomaly_scores.min() + 1e-9)
    dataset["final_risk_score"] = (
        (dataset["ml_incident_probability"] * 100 * 0.60)
        + (dataset["rule_based_score"] * 0.25)
        + (dataset["anomaly_score"] * 0.15)
    ).round(2)
    dataset["risk_level"] = pd.cut(
        dataset["final_risk_score"],
        bins=[-np.inf, 35, 70, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    preprocessor_fitted = deployment_model.named_steps["preprocessor"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = deployment_model.named_steps["classifier"].feature_importances_
    feature_importances = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    feature_importances["feature"] = feature_importances["feature"].str.replace("categorical__", "", regex=False).str.replace(
        "numeric__", "", regex=False
    )

    dataset.to_csv(paths.risk_trends, index=False)
    department_summary, system_summary = build_summary_tables(dataset, paths=paths)

    metrics_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records_used": int(len(dataset)),
        "positive_incident_rate": round(float(dataset["incident_next_7d"].mean()), 4),
        "ml_model": ml_metrics,
        "rule_based_baseline": rule_metrics,
        "anomaly_detector": anomaly_metrics,
        "top_features": feature_importances.to_dict(orient="records"),
        "latest_snapshot_date": str(dataset["date"].max().date()),
        "high_risk_systems_latest": int((system_summary["risk_level"] == "High").sum()),
        "high_risk_departments_latest": int((department_summary["highest_risk_level"] == "High").sum()),
    }

    bundle = {
        "classifier": deployment_model,
        "anomaly_transformer": deployment_anomaly_transformer,
        "anomaly_model": deployment_anomaly_model,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "metrics": metrics_payload,
    }

    ensure_parent(paths.model_bundle)
    ensure_parent(paths.model_metrics)
    joblib.dump(bundle, paths.model_bundle)
    paths.model_metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


def run_pipeline(days: int = 180, seed: int = 42, paths: PipelinePaths = PIPELINE_PATHS) -> dict[str, object]:
    raw_logs = generate_raw_logs(output_path=paths.raw_logs, days=days, seed=seed)
    preprocess_logs(raw_path=paths.raw_logs, output_path=paths.cleaned_logs, seed=seed)
    metrics = train_and_score(cleaned_path=paths.cleaned_logs, paths=paths, seed=seed)
    metrics["raw_log_rows"] = int(len(raw_logs))
    return metrics
