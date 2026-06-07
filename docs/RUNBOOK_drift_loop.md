# Runbook — Drift → Gate loop (`drift_loop.py`)

**Use when:** running the closed-loop drift check, either on a schedule or ad hoc, to decide whether live traffic has drifted enough to warrant promoting a challenger.
**Outcome:** a single decision — do nothing, investigate, or promote — backed by a drift report and a gate report.
**Time:** ~1 minute (with a small `HOLDOUT_N`).

This loop only **detects, gates, and recommends**. It never trains a model and never promotes one — promotion stays a deliberate human action (see `RUNBOOK_promote.md`).

---

## What it chains

```
src.ops.drift_report  --exit code-->  0 stop | 1 abort | 2 drift
                                                          |
                                          src.ops.compare_models (the gate)
                                                          |
                                          read reports/champion_vs_challenger.json
                                                          |
                                          PROMOTE -> recommend | HOLD -> investigate
```

- **Drift** = `monitoring/reference_data.csv` (training baseline) vs `logs/predictions.csv` (live traffic).
- **Gate** = `@champion` vs `@challenger` on a stratified holdout.
- **Run record** = `reports/drift_loop.json` (timestamp, drift result, gate decision, exit code).

---

## Prerequisites

- MLflow tracking server running on `http://127.0.0.1:5000` (the gate loads models from the registry).
- `logs/predictions.csv` exists and has recent traffic (the drift check needs a "current" window).
- The `[monitoring]` extra is installed (`pip install -e ".[monitoring]"`) so Evidently is available.
- Environment set in this shell:
  ```powershell
  $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
  ```

---

## Procedure

1. **Run the loop**
   ```powershell
   python -m src.ops.drift_loop
   echo $LASTEXITCODE
   ```
   (Optional, for a fast gate: `$env:HOLDOUT_N = "50"` before running. Unset with `$env:HOLDOUT_N = ""` to return to the default 500.)

2. **Read the exit code** — this is the verdict a scheduler/CI branches on:

   | Exit | Meaning | Action |
   |---|---|---|
   | `0` | No drift | Nothing. The world still matches training data. |
   | `2` | Drift **and** gate says PROMOTE | A registered challenger is meaningfully better. Follow `RUNBOOK_promote.md` to promote it. |
   | `3` | Drift **but** gate says HOLD | The world moved, but the current `@challenger` is **not** better. Notify the data scientist to train + register a fresh challenger on recent data; do **not** promote. |
   | `1` | A step failed | The drift check or gate crashed. **Do not act on the signal.** Fix the pipeline (see Troubleshooting), then re-run. |

3. **Keep the run record (optional)** — to preserve the audit trail of a loop run:
   ```powershell
   git add reports/drift_loop.json
   git commit -m "Drift loop run: <decision>"
   ```

---

## Scheduling

The loop is exit-code aware so it can run unattended. Suggested cadence: hourly or daily, depending on traffic volume. A scheduler (Windows Task Scheduler / cron / CI) branches on the exit code:

- `0` → log and move on.
- `2` → page the on-call MLOps engineer to run the promotion runbook.
- `3` → notify the data-science channel (challenger needed).
- `1` → alert: the monitoring pipeline itself is broken.

> Exit `1` must page someone. A silently-broken drift check looks identical to "no drift" if nobody is watching the exit code.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loop exits `1` at the **drift** step | `logs/predictions.csv` missing/empty, or Evidently not installed | Send `/predict` traffic first; `pip install -e ".[monitoring]"` |
| Loop exits `1` at the **gate** step | MLflow server down, or `MLFLOW_TRACKING_URI` not set in this shell | Start the server; `$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"` |
| Gate runs for minutes | `HOLDOUT_N` too large (model predicts per-row) | `$env:HOLDOUT_N = "50"` for a fast run; real fix is the batch-predict path |
| Drift verdict flips between runs on the same data | Small "current" window — drift is sample-size sensitive | Use a larger/representative traffic window; tune `DRIFT_SHARE_THRESHOLD` |
| Exit `3` every time with `agreement_rate` 100% | `@challenger` is identical to `@champion` (placeholder) | Expected until a genuinely different challenger is registered |

---

## When NOT to use this runbook

- **To promote a model:** this loop only *recommends*. Promotion has its own runbook (`RUNBOOK_promote.md`).
- **To train/retrain a challenger:** that is the data scientist's job. The loop's exit `3` is the hand-off signal, not a build step.
- **As the sole safety net:** drift detection is an early warning, not proof of harm. Pair it with a scheduled retrain cadence and with the operational dashboards (Prometheus/Grafana) for latency/error health.
