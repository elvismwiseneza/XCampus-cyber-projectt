# Predictive Cyber Threat Analytics for Higher Education Networks

This project is a lightweight, scholarship-ready prototype for predicting cybersecurity risk on a university network before an incident happens. It uses synthetic campus telemetry so you can demonstrate a realistic workflow without needing access to private institutional data.

## What The System Does

The pipeline monitors five core campus cyber signals plus one privileged-access signal:

- login failures
- suspicious IP access patterns
- unusual file downloads
- new device connections
- phishing email patterns
- privileged access attempts

It then produces:

- a cleaned modeling dataset
- a machine-learning prediction of future incident risk
- an anomaly score for unusual network behavior
- a combined final risk score
- low/medium/high flags for departments and systems
- a Streamlit dashboard for presentation

## Project Structure

- `Data/raw-logs.csv`: synthetic event-level security telemetry
- `Data/cleaned-logs.csv`: engineered daily features by department and system
- `Data/risk-trends.csv`: full scored history used for trend charts
- `Data/department-risk-summary.csv`: latest department-level risk snapshot
- `Data/system-risk-summary.csv`: latest system-level risk snapshot
- `Models/anomaly-model.pkl`: saved model bundle for the classifier and anomaly detector
- `Models/model-metrics.json`: evaluation metrics and top feature importances
- `scripts/pipeline.py`: shared implementation for generation, preprocessing, training, and scoring
- `scripts/generate-logs.py`: raw log generation step
- `scripts/preprocess.py`: feature engineering step
- `scripts/train-model.py`: training and artifact generation step
- `risk-scoring.py`: one-command runner for the full workflow
- `Dashboard/app.py`: Streamlit dashboard

## Step-By-Step Build Explanation

### Step 1: Generate Raw Campus Security Events

Run:

```powershell
python scripts/generate-logs.py --days 180 --seed 42
```

What this does:

- creates synthetic network/security events for several university departments
- assigns events to realistic systems like `finance-erp`, `student-records`, and `research-data-vault`
- simulates risk spikes such as suspicious IP access surges, phishing waves, and unusual download behavior
- stores everything as event-level rows in `Data/raw-logs.csv`

Why it matters academically:

- it creates a reproducible campus telemetry dataset
- it lets you test predictive security ideas without handling real student or staff records

### Step 2: Convert Raw Logs Into Risk Features

Run:

```powershell
python scripts/preprocess.py --seed 42
```

What this does:

- aggregates raw logs by `date`, `department`, and `system`
- counts each security signal category
- calculates behavior ratios such as off-hours activity and external-IP activity
- adds rolling 7-day averages and spike ratios so the model can learn trends, not just static counts
- generates a synthetic `incident_next_7d` label for prediction experiments

Why it matters academically:

- this is the predictive part of the project
- it turns behavior into measurable research variables you can analyze

### Step 3: Train The Risk Models

Run:

```powershell
python scripts/train-model.py --seed 42
```

What this does:

- trains a Random Forest model to predict whether a cyber incident is likely in the next 7 days
- trains an Isolation Forest model to detect anomalous behavior
- evaluates those models against a rule-based baseline
- saves model metrics and top feature importances

Why it matters academically:

- it directly supports research questions like:
  - can machine learning predict compromise windows from behavior?
  - which security events matter most?
  - is anomaly detection better than static thresholds?

### Step 4: Produce Final Risk Scores

This happens automatically during training.

What this does:

- combines predictive ML probability, rule-based score, and anomaly score
- converts that combined score into `Low`, `Medium`, or `High`
- generates separate department and system summaries for the newest snapshot date

Why it matters:

- it gives security teams a simple action-oriented output they can use immediately
- it turns raw cyber telemetry into a decision-support tool

### Step 5: Launch The Dashboard

Run:

```powershell
streamlit run Dashboard/app.py
```

What this does:

- shows department-level and system-level risk tables
- visualizes historical risk trends
- compares predictive ML, anomaly detection, and rule-based performance
- highlights the most important signals driving risk

Why it matters:

- it makes the project demo-friendly for scholarship panels, professors, or judges
- it helps you explain the practical value of your research

## One-Command Run

If you want the full workflow in one command, use:

```powershell
python risk-scoring.py --days 180 --seed 42
```

## Suggested Research Questions

- Can user and system behavior predict elevated cyber risk before compromise?
- Which event categories most strongly correlate with future incidents?
- Does anomaly detection outperform rule-based detection on synthetic university telemetry?
- Which departments show the most persistent risk patterns over time?

## Suggested Presentation Talking Points

- The system is predictive, not only reactive.
- It compares multiple detection paradigms in one workflow.
- It uses explainable features rather than black-box alerting alone.
- It is deployable with simulated or public data, making it accessible for student research.

## Minimal Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```
