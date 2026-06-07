"""Close-the-loop orchestrator: drift -> gate -> promotion recommendation.

Chains the monitoring pieces into ONE decision:
1. run the drift check (src.ops.drift_report); branch on its exit code
2. if drift is detected, run the gate (src.ops.compare_models)
3. read the gate's decision and recommend whether to promote @challenger

By design this RECOMMENDS but does not promote: swapping @champion changes what
production serves, so pulling that trigger stays a deliberate human action
(run src.ops.promote yourself -- see docs/RUNBOOK_promote.md). Mirrors the
continuous-delivery rule: automate up to the gate, require approval to ship.

Exit code = the on-call verdict (so cron / Task Scheduler / CI can branch on it):
0 = all clear   - no drift, nothing to do
2 = ACT         - drift detected AND gate recommends promoting @challenger
3 = INVESTIGATE - drift detected but gate says HOLD (challenger not better)
1 = a step FAILED (drift or gate job crashed) - alert, do NOT act

Prereqs: MLflow server up + MLFLOW_TRACKING_URI set (the gate needs the registry),
and logs/predictions.csv populated (the drift check needs live traffic).

Run:  python -m src.ops.drift_loop
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- drift_report.py's exit-code contract (defined in that file) ---
DRIFT_NONE = 0
DRIFT_DETECTED = 2

# --- this orchestrator's own exit codes (what a scheduler branches on) ---
LOOP_OK = 0           # no drift
LOOP_FAILED = 1       # a sub-step crashed
LOOP_ACT = 2          # drift + PROMOTE recommended
LOOP_INVESTIGATE = 3  # drift + HOLD

GATE_REPORT = Path("reports/champion_vs_challenger.json")
LOOP_REPORT = Path("reports/drift_loop.json")


def _run_step(title: str, module: str) -> int:
    """Run a pipeline step as a subprocess and return its exit code.

    We shell out (instead of importing) so each step stays an independent tool
    with an exit-code/artifact contract -- the same way Airflow/cron/CI run tasks.
    sys.executable = the exact Python/venv running us (portable; not bare 'python').
    No capture_output, so the child's console output streams straight through.
    """
    print(f"\n=== {title} ===")
    return subprocess.run([sys.executable, "-m", module], check=False).returncode


def _write_summary(summary: dict) -> None:
    """Persist an auditable run record (git-diffable, like the other reports/*.json)."""
    LOOP_REPORT.parent.mkdir(parents=True, exist_ok=True)
    LOOP_REPORT.write_text(json.dumps(summary, indent=2))
    print(f"[loop] Run summary -> {LOOP_REPORT}")


def main() -> int:
    summary = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # --- Step 1: drift check -------------------------------------------------
    drift_code = _run_step("DRIFT CHECK", "src.ops.drift_report")

    if drift_code == DRIFT_NONE:
        print("\n[loop] No drift. Nothing to do.")
        summary.update(drift=False, decision=None, exit=LOOP_OK)
        _write_summary(summary)
        return LOOP_OK

    if drift_code != DRIFT_DETECTED:
        # Not 0 and not 2 -> the drift job itself failed (exit 1). Never run the
        # gate or recommend a promote off a broken signal -- abort cleanly.
        print(f"\n[loop] Drift check FAILED (exit {drift_code}). Aborting, no action.")
        summary.update(drift=None, decision=None, exit=LOOP_FAILED)
        _write_summary(summary)
        return LOOP_FAILED

    # --- Step 2: drift detected -> run the gate ------------------------------
    # NOTE: the gate compares against whatever @challenger exists NOW. In a real
    # loop the data scientist first retrains on fresh labeled data and registers
    # a new @challenger -- that handoff is human, not ours.
    print("\n[loop] Drift detected -> running champion/challenger gate.")
    gate_code = _run_step("GATE", "src.ops.compare_models")
    if gate_code != 0:
        print(f"\n[loop] Gate FAILED (exit {gate_code}). Aborting, no action.")
        summary.update(drift=True, decision=None, exit=LOOP_FAILED)
        _write_summary(summary)
        return LOOP_FAILED

    # --- Step 3: read the gate's decision ARTIFACT (it has no exit-code verdict) ---
    gate = json.loads(GATE_REPORT.read_text())
    decision = gate["decision"]  # "PROMOTE challenger" | "HOLD champion"
    summary.update(drift=True, decision=decision, gate=gate)

    if decision.startswith("PROMOTE"):
        print("\n[loop] Gate recommends PROMOTE.")
        print("       Review reports/champion_vs_challenger.json, then run:")
        print("         python -m src.ops.promote --to-champion <CHALLENGER_VERSION>")
        print("       (promotion stays manual -- see docs/RUNBOOK_promote.md)")
        summary["exit"] = LOOP_ACT
        _write_summary(summary)
        return LOOP_ACT

    print("\n[loop] Drift detected but gate says HOLD (challenger not better).")
    print("       Investigate: a fresh challenger may need to be trained & registered.")
    summary["exit"] = LOOP_INVESTIGATE
    _write_summary(summary)
    return LOOP_INVESTIGATE


if __name__ == "__main__":
    sys.exit(main())  # propagate the verdict as the process exit code