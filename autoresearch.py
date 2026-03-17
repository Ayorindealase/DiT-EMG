"""
autoresearch.py — Autonomous Research Loop for DiT-EMG
=======================================================
Runs overnight. Calls Claude API to iteratively improve train.py,
guided by program.md and the results from each experiment.

Usage:
    # In Kaggle notebook:
    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
    !python autoresearch.py --experiments 50

    # Or locally:
    export ANTHROPIC_API_KEY=sk-ant-...
    python autoresearch.py --experiments 20
"""

import os
import sys
import json
import time
import shutil
import argparse
import requests
import subprocess
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ANTHROPIC_API_URL  = "https://api.anthropic.com/v1/messages"
MODEL              = "claude-sonnet-4-20250514"
MAX_TOKENS         = 4096

TRAIN_FILE         = Path("train.py")
PROGRAM_FILE       = Path("program.md")
RESULTS_FILE       = Path("results.jsonl")
EXPERIMENT_LOG     = Path("experiment_log.md")
BACKUP_DIR         = Path("backups")

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.\n"
            "In Kaggle: from kaggle_secrets import UserSecretsClient; "
            "os.environ['ANTHROPIC_API_KEY'] = UserSecretsClient().get_secret('ANTHROPIC_API_KEY')\n"
            "Locally: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return key


def read_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def read_results(n_recent: int = 10) -> str:
    """Read the last N experiment results from results.jsonl."""
    if not RESULTS_FILE.exists():
        return "No results yet — this is the first experiment."

    lines = RESULTS_FILE.read_text().strip().split("\n")
    lines = [l for l in lines if l.strip()]

    if not lines:
        return "No results yet — this is the first experiment."

    recent = lines[-n_recent:]
    results = []
    for line in recent:
        try:
            r = json.loads(line)
            results.append(
                f"  step={r.get('step','?')} | "
                f"val_fid={r.get('fid', r.get('val_fid','?')):.4f} | "
                f"tstr_acc={r.get('tstr_acc','?'):.4f} | "
                f"psd_error={r.get('psd_error','?'):.4f} | "
                f"elapsed={r.get('elapsed_s','?'):.0f}s"
            )
        except Exception:
            continue

    if not results:
        return "No valid results yet."

    # Find best FID
    best_fid = float("inf")
    for line in lines:
        try:
            r = json.loads(line)
            fid = r.get('fid', r.get('val_fid', float('inf')))
            if isinstance(fid, (int, float)) and fid < best_fid:
                best_fid = fid
        except Exception:
            continue

    return (
        f"Best val_fid so far: {best_fid:.4f}\n"
        f"Last {len(results)} experiments:\n" +
        "\n".join(results)
    )


def read_experiment_log(n_recent: int = 3) -> str:
    """Read the last N experiment log entries."""
    if not EXPERIMENT_LOG.exists():
        return "No experiment log yet."

    content = EXPERIMENT_LOG.read_text(encoding="utf-8")
    # Split by experiment headers
    entries = content.split("## Experiment")
    entries = [e for e in entries if e.strip()]

    if not entries:
        return "No experiment log yet."

    recent = entries[-n_recent:]
    return "## Experiment" + "\n## Experiment".join(recent)


def backup_train_py(experiment_n: int):
    """Save a backup of train.py before modifying."""
    BACKUP_DIR.mkdir(exist_ok=True)
    backup_path = BACKUP_DIR / f"train_exp{experiment_n:03d}.py"
    shutil.copy(TRAIN_FILE, backup_path)


def restore_train_py(experiment_n: int) -> bool:
    """Restore train.py from backup if experiment failed."""
    backup_path = BACKUP_DIR / f"train_exp{experiment_n:03d}.py"
    if backup_path.exists():
        shutil.copy(backup_path, TRAIN_FILE)
        print(f"  [restore] train.py restored from backup {experiment_n}")
        return True
    return False


# ─────────────────────────────────────────────
# CLAUDE API CALL
# ─────────────────────────────────────────────

def call_claude(prompt: str, system: str, api_key: str) -> str:
    """Call Claude API and return the response text."""
    headers = {
        "Content-Type":      "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key":         api_key,
    }
    body = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     system,
        "messages":   [{"role": "user", "content": prompt}],
    }

    for attempt in range(3):
        try:
            resp = requests.post(ANTHROPIC_API_URL, headers=headers,
                                 json=body, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                return data["content"][0]["text"]
            elif resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [api] Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [api] Error {resp.status_code}: {resp.text[:200]}")
                time.sleep(10)
        except Exception as e:
            print(f"  [api] Exception: {e}")
            time.sleep(10)

    raise RuntimeError("Claude API call failed after 3 attempts")


# ─────────────────────────────────────────────
# AGENT PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an elite ML research agent working on DiT-EMG — 
a Diffusion Transformer for synthetic sEMG generation. 

Your sole job is to improve val_fid by making ONE precise, 
theoretically-justified change to train.py per experiment.

You must respond in this EXACT format and nothing else:

HYPOTHESIS: <one sentence — why this change should improve val_fid>

CHANGE: <exact description of what to modify>

NEW_TRAIN_PY:
```python
<complete modified train.py — the entire file>
```

EXPERIMENT_LOG_ENTRY:
```markdown
## Experiment {N} — <one-line description>

**Theoretical motivation**: <why this works based on EMG signal properties>

**Change**: <exact change made>

**Prediction**: <expected direction and magnitude on val_fid>
```

Rules:
- Change ONLY the hyperparameters block or specific functions — never change the evaluate() call or TRAIN_TIME_SECONDS
- One change per experiment — never modify two independent things
- The complete train.py must be syntactically valid Python
- Base your decision on the results and experiment log provided
"""


def build_prompt(experiment_n: int, train_py: str,
                  results_summary: str, recent_log: str,
                  program: str) -> str:
    return f"""You are running Experiment {experiment_n}.

## Research Brief (program.md summary)
{program[:2000]}

## Current Results
{results_summary}

## Recent Experiment Log
{recent_log}

## Current train.py
```python
{train_py}
```

Based on the results and your research brief, decide what single change 
to make for Experiment {experiment_n}. 

Remember:
- val_fid baseline is 3218.9959 (Experiment 1 on real NinaPro E2 data)
- Target: val_fid < 500, tstr_acc/trtr_acc ratio > 0.85
- Make ONE theoretically-justified change
- Output the COMPLETE modified train.py

Respond now as the elite researcher you are."""


# ─────────────────────────────────────────────
# PARSE CLAUDE RESPONSE
# ─────────────────────────────────────────────

def parse_response(response: str) -> dict:
    """Extract hypothesis, new train.py, and log entry from Claude's response."""
    result = {
        "hypothesis":    "",
        "change":        "",
        "new_train_py":  "",
        "log_entry":     "",
    }

    # Extract HYPOTHESIS
    if "HYPOTHESIS:" in response:
        hyp_start = response.index("HYPOTHESIS:") + len("HYPOTHESIS:")
        hyp_end   = response.index("\n", hyp_start)
        result["hypothesis"] = response[hyp_start:hyp_end].strip()

    # Extract CHANGE
    if "CHANGE:" in response:
        chg_start = response.index("CHANGE:") + len("CHANGE:")
        chg_end   = response.index("\n", chg_start)
        result["change"] = response[chg_start:chg_end].strip()

    # Extract new train.py (between ```python and ```)
    if "NEW_TRAIN_PY:" in response and "```python" in response:
        py_start = response.index("```python", response.index("NEW_TRAIN_PY:")) + len("```python")
        py_end   = response.index("```", py_start)
        result["new_train_py"] = response[py_start:py_end].strip()

    # Extract experiment log entry
    if "EXPERIMENT_LOG_ENTRY:" in response and "```markdown" in response:
        log_start = response.index("```markdown", response.index("EXPERIMENT_LOG_ENTRY:")) + len("```markdown")
        log_end   = response.index("```", log_start)
        result["log_entry"] = response[log_start:log_end].strip()

    return result


def validate_train_py(code: str) -> bool:
    """Check that the new train.py is valid Python."""
    import ast
    try:
        ast.parse(code)
        # Check critical constraints
        if "TRAIN_TIME_SECONDS" not in code:
            print("  [validate] WARN: TRAIN_TIME_SECONDS missing")
            return False
        if "evaluate(" not in code:
            print("  [validate] WARN: evaluate() call missing")
            return False
        return True
    except SyntaxError as e:
        print(f"  [validate] Syntax error: {e}")
        return False


# ─────────────────────────────────────────────
# RUN ONE EXPERIMENT
# ─────────────────────────────────────────────

def run_training() -> dict:
    """Run python train.py and capture the final metrics."""
    print("  [train] Running python train.py (5 min budget)...")
    start = time.time()

    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True,
        text=True,
        timeout=450,  # 7.5 min timeout (generous buffer)
    )

    elapsed = time.time() - start
    output  = result.stdout + result.stderr

    # Parse final metrics from output
    metrics = {"elapsed_s": elapsed, "success": result.returncode == 0}

    if result.returncode != 0:
        print(f"  [train] FAILED (returncode={result.returncode})")
        print(f"  [train] Last 500 chars: {output[-500:]}")
        return metrics

    # Extract metrics from the Final Result block
    lines = output.split("\n")
    for line in lines:
        line = line.strip()
        for key in ["val_fid", "fid", "tstr_acc", "trtr_acc",
                    "tstr_f1", "psd_error", "dtw_mean"]:
            if f"{key}" in line and ":" in line:
                try:
                    val = float(line.split(":")[-1].strip().split()[0])
                    metrics[key] = val
                except Exception:
                    pass

    # Normalise fid key
    if "val_fid" in metrics and "fid" not in metrics:
        metrics["fid"] = metrics["val_fid"]

    print(f"  [train] Done in {elapsed:.0f}s — "
          f"val_fid={metrics.get('fid', metrics.get('val_fid', '?'))}")

    return metrics


def append_log_entry(entry: str, experiment_n: int, metrics: dict):
    """Append result metrics to the experiment log entry."""
    result_block = (
        f"\n**Result**:\n"
        f"- val_fid:   {metrics.get('fid', metrics.get('val_fid', '?'))}\n"
        f"- tstr_acc:  {metrics.get('tstr_acc', '?')}\n"
        f"- trtr_acc:  {metrics.get('trtr_acc', '?')}\n"
        f"- psd_error: {metrics.get('psd_error', '?')}\n"
        f"- dtw_mean:  {metrics.get('dtw_mean', '?')}\n"
    )

    full_entry = entry + result_block + "\n\n---\n\n"

    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(full_entry)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run_autoresearch(n_experiments: int = 50, start_from: int = 2):
    """
    Main autoresearch loop.
    Runs n_experiments iterations of: Claude decides → modify train.py → train → evaluate → repeat
    """
    api_key = get_api_key()
    program = read_file(PROGRAM_FILE)

    print(f"\n{'='*60}")
    print(f"  DiT-EMG Autoresearch Loop")
    print(f"  Model : {MODEL}")
    print(f"  Experiments : {n_experiments} (starting from {start_from})")
    print(f"  Baseline val_fid : 3218.9959")
    print(f"{'='*60}\n")

    # Track best
    best_fid = 3218.9959  # baseline from Experiment 1
    session_start = time.time()

    for exp_n in range(start_from, start_from + n_experiments):
        print(f"\n{'─'*50}")
        print(f"  EXPERIMENT {exp_n}")
        print(f"  Time elapsed: {(time.time()-session_start)/3600:.1f}h")
        print(f"  Best FID so far: {best_fid:.4f}")
        print(f"{'─'*50}")

        # ── Read current state ────────────────────────────
        train_py        = read_file(TRAIN_FILE)
        results_summary = read_results(n_recent=8)
        recent_log      = read_experiment_log(n_recent=3)

        # ── Ask Claude what to change ─────────────────────
        print(f"  [claude] Asking for Experiment {exp_n} decision...")
        prompt = build_prompt(
            experiment_n    = exp_n,
            train_py        = train_py,
            results_summary = results_summary,
            recent_log      = recent_log,
            program         = program,
        )

        try:
            response = call_claude(prompt, SYSTEM_PROMPT, api_key)
        except Exception as e:
            print(f"  [claude] API call failed: {e}")
            print("  Skipping this experiment...")
            time.sleep(30)
            continue

        # ── Parse response ────────────────────────────────
        parsed = parse_response(response)

        print(f"  [claude] Hypothesis: {parsed['hypothesis']}")
        print(f"  [claude] Change: {parsed['change']}")

        if not parsed["new_train_py"]:
            print("  [parse] Could not extract new train.py — skipping")
            continue

        # ── Validate and apply change ──────────────────────
        if not validate_train_py(parsed["new_train_py"]):
            print("  [validate] Invalid train.py — skipping experiment")
            continue

        # Backup current train.py
        backup_train_py(exp_n)

        # Write new train.py
        TRAIN_FILE.write_text(parsed["new_train_py"], encoding="utf-8")
        print(f"  [apply] train.py updated")

        # ── Run training ──────────────────────────────────
        metrics = run_training()

        if not metrics["success"]:
            print(f"  [result] Training failed — reverting train.py")
            restore_train_py(exp_n)
            if parsed["log_entry"]:
                append_log_entry(
                    parsed["log_entry"] + "\n**Decision**: REVERTED — training crashed",
                    exp_n, metrics
                )
            continue

        # ── Evaluate result ───────────────────────────────
        current_fid = metrics.get("fid", metrics.get("val_fid", float("inf")))

        if isinstance(current_fid, (int, float)) and current_fid < best_fid:
            improvement = (best_fid - current_fid) / best_fid * 100
            print(f"  [result] ✓ IMPROVEMENT: {best_fid:.4f} → {current_fid:.4f} "
                  f"(-{improvement:.1f}%)")
            best_fid = current_fid
            decision = "KEEP"
        else:
            print(f"  [result] ✗ No improvement: {current_fid} vs best {best_fid:.4f}")
            print(f"  [result] Reverting train.py...")
            restore_train_py(exp_n)
            decision = "REVERTED"

        # ── Log result ────────────────────────────────────
        if parsed["log_entry"]:
            append_log_entry(
                parsed["log_entry"] + f"\n**Decision**: {decision}",
                exp_n, metrics
            )

        # ── Print summary ──────────────────────────────────
        print(f"\n  Experiment {exp_n} summary:")
        print(f"  Decision  : {decision}")
        print(f"  val_fid   : {current_fid}")
        print(f"  Best so far: {best_fid:.4f}")

        # Small pause between experiments
        time.sleep(5)

    # ── Final summary ─────────────────────────────────────
    total_time = (time.time() - session_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Autoresearch complete!")
    print(f"  Total time    : {total_time:.1f} hours")
    print(f"  Experiments   : {n_experiments}")
    print(f"  Starting FID  : 3218.9959")
    print(f"  Final best FID: {best_fid:.4f}")
    print(f"  Improvement   : {(3218.9959 - best_fid) / 3218.9959 * 100:.1f}%")
    print(f"{'='*60}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Experiment log : {EXPERIMENT_LOG}")
    print(f"Best model     : checkpoints/best_model.pt")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT-EMG Autoresearch Loop")
    parser.add_argument("--experiments", type=int, default=50,
                        help="Number of experiments to run (default: 50)")
    parser.add_argument("--start-from",  type=int, default=2,
                        help="Start experiment number (default: 2, after baseline)")
    args = parser.parse_args()

    run_autoresearch(
        n_experiments = args.experiments,
        start_from    = args.start_from,
    )