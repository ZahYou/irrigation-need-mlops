# Runbook — Promote a new model version to `@champion`

  **Use when:** a data scientist has registered a new version of `irrigation-need-classifier` and tagged it `@challenger`.
  **Outcome:** the new version becomes `@champion`; the old champion is preserved as `@previous` for instant rollback.
  **Time:** ~5 minutes.

  ---

  ## Prerequisites

  - MLflow tracking server is running on `http://127.0.0.1:5000` (or the prod URL).
  - The new version exists in the registry and is tagged `@challenger`.
  - You can reach the server from your shell:
    ```powershell
    $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    python -m src.ops.promote --status
    Should print current aliases without error.

  ---
  Procedure

  1. Inspect current state

  python -m src.ops.promote --status

  Confirm @champion and @challenger are both set, and @challenger points at the version you intend to promote.

  2. Run the comparison gate

  python -m src.ops.compare_models

  Produces reports/champion_vs_challenger.json. Read the decision field:

  ┌────────────────────┬─────────────────────────────────────────────────┐
  │      Decision      │                     Action                      │
  ├────────────────────┼─────────────────────────────────────────────────┤
  │ PROMOTE challenger │ Proceed to step 3                               │
  ├────────────────────┼─────────────────────────────────────────────────┤
  │ HOLD champion      │ STOP. Notify the data scientist; do not promote │
  └────────────────────┴─────────────────────────────────────────────────┘

  Also sanity-check agreement_rate. If it's 100%, the challenger is identical to champion — promoting is a no-op (still safe, but ask whether this
   was intended).

  3. Promote

  python -m src.ops.promote --to-champion <VERSION>

  The CLI will:
  - Refuse if <VERSION> is not currently @challenger (override with --force)
  - Load the target model and run a smoke prediction
  - Move the existing @champion to @previous
  - Set the new @champion

  4. Restart the serving app

  The container loads the model in lifespan (at startup only). It won't see the new champion until restarted.

  docker ps                         # find the container ID
  docker restart <CONTAINER_ID>

  Watch the logs:
  docker logs -f <CONTAINER_ID>
  You should see: Loaded model: models:/irrigation-need-classifier@champion.

  5. Verify serving

  Invoke-RestMethod http://127.0.0.1:8000/health
  Expected: loaded : True.

  Send a known-good payload to /predict and confirm the response is well-formed (see tests/test_api.py for examples).

  6. Commit the gate report

  git add reports/champion_vs_challenger.json
  git commit -m "Promote V<NEW> to @champion (gate report)"

  ---
  Rollback

  If the new champion is misbehaving in production:

  python -m src.ops.promote --rollback
  docker restart <CONTAINER_ID>

  This swaps @champion ↔ @previous. Total time: ~10 seconds. No code change, no rebuild.

  ---
  Troubleshooting

  ┌───────────────────────────────────────────┬────────────────────────────────────────┬─────────────────────────────────────────────────────┐
  │                  Symptom                  │              Likely cause              │                         Fix                         │
  ├───────────────────────────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ ConnectionRefusedError on any python -m   │ MLflow tracking server not running     │ Start it: see summary2.md → "How to run"            │
  │ src.ops.* command                         │                                        │                                                     │
  ├───────────────────────────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ mlflow-artifacts URI was supplied, but    │ MLFLOW_TRACKING_URI not set in this    │ $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  │
  │ the tracking URI was sqlite:/...          │ terminal                               │                                                     │
  ├───────────────────────────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Could not find a registered artifact      │ Tracking server was started with wrong │ Kill stale servers; relaunch from project root with │
  │ repository for: c:                        │  --artifacts-destination               │  --artifacts-destination ./mlartifacts              │
  ├───────────────────────────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Smoke prediction fails during promote     │ Target model artifact broken or        │ Do not override. Reject the promotion; notify the   │
  │                                           │ schema-incompatible                    │ data scientist with the traceback                   │
  ├───────────────────────────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ /health returns loaded: False after       │ Container can't reach the tracking     │ Confirm host.docker.internal:5000 is reachable;     │
  │ restart                                   │ server                                 │ check --allowed-hosts includes it                   │
  ├───────────────────────────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Gate runs for >5 minutes                  │ Holdout too large (model is per-row)   │ Reduce HOLDOUT_N in compare_models.py               │
  └───────────────────────────────────────────┴────────────────────────────────────────┴─────────────────────────────────────────────────────┘

  ---
  When NOT to use this runbook

  - Emergency hotfix without a registered challenger: use --force and document why in the commit message.
  - Rolling back across multiple versions: --rollback only swaps one step. For deeper rollback, set @champion manually via the MLflow UI and
  document it.
  - Schema-breaking change: if the new version's input schema differs from the old, the serving app's Pydantic model must change too. This is a
  code deploy, not just a promotion. Coordinate with the app owner.