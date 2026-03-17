"""
autoresearch.py — Autonomous Research Loop for DiT-EMG
=======================================================
Calls Claude API to iteratively improve train.py overnight.

Key design: Claude returns ONE line change (old line → new line).
We apply it with a simple string replace. No full-file rewrites.
This is reliable, fast, and never fails to parse.

Usage:
    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
    !python autoresearch.py --experiments 30 --start-from 2
"""

import os
import sys
import re
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

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-sonnet-4-20250514"
MAX_TOKENS        = 1024   # Small — we only need 4 lines back, not 833

TRAIN_FILE        = Path("train.py")
PROGRAM_FILE      = Path("program.md")
RESULTS_FILE      = Path("results.jsonl")
EXPERIMENT_LOG    = Path("experiment_log.md")
PAPER_TRACKER     = Path("paper_tracker.md")
BACKUP_DIR        = Path("backups")

BASELINE_FID      = 3218.9959   # Experiment 1 — real NinaPro E2 data

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: import os; from kaggle_secrets import UserSecretsClient; "
            "os.environ['ANTHROPIC_API_KEY'] = UserSecretsClient().get_secret('ANTHROPIC_API_KEY')"
        )
    return key


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def extract_hparams_block(train_py: str) -> str:
    """Extract only the hyperparameters block from train.py."""
    lines      = train_py.split("\n")
    collecting = False
    result     = []

    for line in lines:
        if "HYPERPARAMETERS" in line and "❶" in line:
            collecting = True
        if collecting:
            result.append(line)
        if collecting and len(result) > 5 and line.strip() == "":
            # Stop at first blank line after 5+ lines collected
            if len(result) > 30:
                break

    # Fallback — just grab lines with the key hyperparameter names
    if len(result) < 5:
        keywords = ["TRAIN_TIME_SECONDS", "BATCH_SIZE", "LEARNING_RATE",
                    "WEIGHT_DECAY", "PATCH_SIZE", "D_MODEL", "N_HEADS",
                    "DEPTH", "D_FF_MULT", "DROPOUT", "T_STEPS", "SCHEDULE",
                    "BETA_START", "BETA_END", "SAMPLE_STEPS", "SAMPLE_GUIDANCE",
                    "CFG_DROPOUT", "CLASS_EMBED_DIM"]
        result = [l for l in lines if any(k in l for k in keywords)]

    return "\n".join(result[:60])


def read_results_summary(n: int = 8) -> str:
    """Summarise last N results from results.jsonl."""
    if not RESULTS_FILE.exists():
        return f"No results yet. Baseline val_fid = {BASELINE_FID}"

    lines = [l for l in RESULTS_FILE.read_text().split("\n") if l.strip()]
    if not lines:
        return f"No results yet. Baseline val_fid = {BASELINE_FID}"

    best_fid = BASELINE_FID
    rows     = []
    for line in lines[-n:]:
        try:
            r   = json.loads(line)
            fid = r.get("fid", r.get("val_fid", "?"))
            if isinstance(fid, float) and fid < best_fid:
                best_fid = fid
            rows.append(
                f"  fid={fid:.2f}  tstr={r.get('tstr_acc','?'):.3f}  "
                f"psd={r.get('psd_error','?'):.1f}  step={r.get('step','?')}"
            )
        except Exception:
            continue

    return (
        f"Baseline val_fid : {BASELINE_FID}\n"
        f"Best val_fid so far: {best_fid:.4f}\n"
        f"Last {len(rows)} results:\n" + "\n".join(rows)
    )


def read_recent_log(n: int = 2) -> str:
    """Read last N experiment log entries."""
    if not EXPERIMENT_LOG.exists():
        return "No experiments yet."
    entries = EXPERIMENT_LOG.read_text(encoding="utf-8").split("---")
    recent  = [e.strip() for e in entries if e.strip()][-n:]
    return "\n\n---\n\n".join(recent) if recent else "No experiments yet."


def backup_train_py(exp_n: int):
    BACKUP_DIR.mkdir(exist_ok=True)
    shutil.copy(TRAIN_FILE, BACKUP_DIR / f"train_exp{exp_n:03d}.py")


def restore_train_py(exp_n: int):
    backup = BACKUP_DIR / f"train_exp{exp_n:03d}.py"
    if backup.exists():
        shutil.copy(backup, TRAIN_FILE)
        print(f"  [restore] train.py restored from experiment {exp_n} backup")


# ─────────────────────────────────────────────
# CLAUDE API
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an elite ML research agent improving DiT-EMG — a diffusion transformer \
for synthetic sEMG generation. Your job: make ONE precise, theoretically-justified \
change to reduce val_fid.

You MUST respond in EXACTLY this format — four labeled fields, nothing else:

HYPOTHESIS: <one sentence: why this specific change reduces val_fid based on EMG signal theory>

CHANGE_LINE: <the exact current line from train.py to change — copy it verbatim>

NEW_LINE: <the exact replacement line — only the line itself, no explanation>

LOG_ENTRY: <2-3 sentences: theoretical motivation, predicted effect, what metric to watch>

Rules:
- ONE change only — never modify two independent parameters
- CHANGE_LINE must be an exact verbatim copy of a line that exists in the hyperparameters
- Do not include any code blocks, backticks, or extra text
- Base decisions on EMG signal theory, not random search
"""


def build_prompt(exp_n: int, hparams: str,
                 results: str, recent_log: str) -> str:
    return f"""\
Experiment {exp_n}.

CURRENT RESULTS:
{results}

RECENT EXPERIMENT LOG:
{recent_log}

CURRENT HYPERPARAMETERS (from train.py):
{hparams}

What single change should Experiment {exp_n} test?
Remember: target val_fid < 500, tstr_acc/trtr_acc ratio > 0.85.
Respond with HYPOTHESIS, CHANGE_LINE, NEW_LINE, LOG_ENTRY only.\
"""


def call_claude(prompt: str, api_key: str) -> str:
    headers = {
        "Content-Type":      "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key":         api_key,
    }
    body = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": prompt}],
    }
    for attempt in range(3):
        try:
            resp = requests.post(ANTHROPIC_API_URL, headers=headers,
                                 json=body, timeout=60)
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"]
            elif resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [api] Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [api] HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(10)
        except Exception as e:
            print(f"  [api] Error: {e}")
            time.sleep(10)
    raise RuntimeError("Claude API failed after 3 attempts")


# ─────────────────────────────────────────────
# PARSE RESPONSE
# ─────────────────────────────────────────────

def parse_response(response: str) -> dict:
    """
    Parse Claude's response. Expects exactly:
        HYPOTHESIS: ...
        CHANGE_LINE: ...
        NEW_LINE: ...
        LOG_ENTRY: ...
    """
    result = {
        "hypothesis":  "",
        "change_line": "",
        "new_line":    "",
        "log_entry":   "",
    }

    # Split response into lines for clean parsing
    lines = response.strip().split("\n")

    current_field = None
    current_value = []

    field_map = {
        "HYPOTHESIS:":  "hypothesis",
        "CHANGE_LINE:": "change_line",
        "NEW_LINE:":    "new_line",
        "LOG_ENTRY:":   "log_entry",
    }

    for line in lines:
        stripped = line.strip()

        # Check if this line starts a new field
        matched_field = None
        for label, key in field_map.items():
            if stripped.startswith(label):
                matched_field = key
                # Save previous field
                if current_field:
                    result[current_field] = " ".join(current_value).strip()
                # Start new field
                current_field = key
                current_value = [stripped[len(label):].strip()]
                break

        if not matched_field and current_field:
            # Continue accumulating multi-line field
            if stripped:
                current_value.append(stripped)

    # Save last field
    if current_field:
        result[current_field] = " ".join(current_value).strip()

    return result


def apply_line_change(change_line: str, new_line: str) -> bool:
    """
    Apply a single line change to train.py.
    Returns True if the line was found and replaced.
    """
    src = read_file(TRAIN_FILE)

    # Strip any accidental surrounding quotes or backticks Claude might add
    change_line = change_line.strip().strip("`").strip("'").strip('"')
    new_line    = new_line.strip().strip("`").strip("'").strip('"')

    if not change_line:
        print("  [apply] Empty change_line — skipping")
        return False

    if change_line not in src:
        # Try fuzzy match — find line containing key part
        key_part = change_line.split("=")[0].strip() if "=" in change_line else change_line[:20]
        candidates = [l for l in src.split("\n") if key_part in l and "=" in l]
        if candidates:
            # Use the closest match
            change_line = candidates[0]
            print(f"  [apply] Fuzzy matched line: {change_line.strip()[:60]}")
        else:
            print(f"  [apply] Line not found: {change_line[:60]}")
            return False

    new_src = src.replace(change_line, new_line, 1)

    # Validate the result is still valid Python
    try:
        import ast
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"  [apply] Syntax error after change: {e}")
        return False

    TRAIN_FILE.write_text(new_src, encoding="utf-8")
    print(f"  [apply] ✓ {change_line.strip()[:55]}")
    print(f"       → {new_line.strip()[:55]}")
    return True


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def run_training() -> dict:
    """Run train.py and return metrics dict."""
    print("  [train] Running train.py (5 min budget)...")
    start = time.time()

    proc = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=450,
    )

    elapsed = time.time() - start
    output  = proc.stdout + proc.stderr
    metrics = {"elapsed_s": elapsed, "success": proc.returncode == 0}

    if proc.returncode != 0:
        print(f"  [train] FAILED — last error:")
        # Print last 3 lines of stderr
        err_lines = [l for l in proc.stderr.split("\n") if l.strip()]
        for l in err_lines[-3:]:
            print(f"    {l}")
        return metrics

    # Parse metrics from output
    for line in output.split("\n"):
        line = line.strip()
        for key in ["val_fid", "fid", "tstr_acc", "trtr_acc",
                    "tstr_f1", "psd_error", "dtw_mean"]:
            if f"{key}" in line and ":" in line:
                try:
                    val = float(line.split(":")[-1].strip().split()[0])
                    metrics[key] = val
                except Exception:
                    pass

    if "val_fid" in metrics and "fid" not in metrics:
        metrics["fid"] = metrics["val_fid"]

    fid = metrics.get("fid", metrics.get("val_fid", "?"))
    print(f"  [train] Done in {elapsed:.0f}s | val_fid={fid}")
    return metrics


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def log_experiment(exp_n: int, hypothesis: str, change_line: str,
                   new_line: str, log_entry: str,
                   metrics: dict, decision: str):
    """Write to experiment_log.md."""
    fid      = metrics.get("fid", metrics.get("val_fid", "?"))
    tstr     = metrics.get("tstr_acc", "?")
    trtr     = metrics.get("trtr_acc", "?")
    psd      = metrics.get("psd_error", "?")
    dtw      = metrics.get("dtw_mean", "?")
    elapsed  = metrics.get("elapsed_s", "?")

    entry = f"""## Experiment {exp_n} — {change_line.strip()[:60]}

**Hypothesis**: {hypothesis}

**Change**:
- Before: `{change_line.strip()}`
- After:  `{new_line.strip()}`

**Agent notes**: {log_entry}

**Result**:
| Metric | Value |
|--------|-------|
| val_fid | {fid} |
| tstr_acc | {tstr} |
| trtr_acc | {trtr} |
| psd_error | {psd} |
| dtw_mean | {dtw} |
| elapsed_s | {elapsed:.0f}s |

**Decision**: {decision}

---

"""
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(entry)


def update_paper_tracker(exp_n: int, change_line: str, new_line: str,
                          metrics: dict, decision: str, best_fid: float):
    """Add a row to the results table in paper_tracker.md."""
    if not PAPER_TRACKER.exists():
        return

    fid      = metrics.get("fid", metrics.get("val_fid", 0))
    tstr_acc = metrics.get("tstr_acc", 0)
    trtr_acc = metrics.get("trtr_acc", 0)
    psd      = metrics.get("psd_error", 0)
    ratio    = round(tstr_acc / trtr_acc, 3) if trtr_acc and trtr_acc > 0 else 0

    # Short description for table
    desc = new_line.strip()[:45] if new_line else change_line.strip()[:45]

    new_row = (
        f"| {exp_n} | {desc} | "
        f"{fid:.2f} | {tstr_acc:.4f} | {trtr_acc:.4f} | "
        f"{ratio:.3f} | {psd:.2f} | {decision} |\n"
    )

    content = PAPER_TRACKER.read_text(encoding="utf-8")
    marker  = "---\n\n## SECTION B"
    if marker in content:
        content = content.replace(marker, new_row + marker, 1)

    # Update timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = re.sub(
        r"\*\*Last updated\*\*:.*",
        f"**Last updated**: {ts} (Experiment {exp_n})",
        content
    )

    # Update best config block if improved
    if decision == "KEEP":
        train_src = read_file(TRAIN_FILE)
        hparams   = {}
        for param in ["D_MODEL", "DEPTH", "N_HEADS", "PATCH_SIZE",
                      "T_STEPS", "SCHEDULE", "SAMPLE_GUIDANCE",
                      "LEARNING_RATE", "BATCH_SIZE"]:
            for line in train_src.split("\n"):
                if param in line and "=" in line and not line.strip().startswith("#"):
                    try:
                        val = line.split("=")[1].strip().split("#")[0].strip().strip('"')
                        hparams[param] = val
                        break
                    except Exception:
                        pass

        best_block = (
            f"```\n"
            f"Best val_fid     : {best_fid:.4f}\n"
            f"Best experiment  : {exp_n}\n"
            f"Key hyperparams  : D_MODEL={hparams.get('D_MODEL','?')}, "
            f"DEPTH={hparams.get('DEPTH','?')}, "
            f"N_HEADS={hparams.get('N_HEADS','?')}, "
            f"PATCH={hparams.get('PATCH_SIZE','?')}\n"
            f"                   T={hparams.get('T_STEPS','?')}, "
            f"Schedule={hparams.get('SCHEDULE','?')}, "
            f"Guidance={hparams.get('SAMPLE_GUIDANCE','?')}\n"
            f"                   Batch={hparams.get('BATCH_SIZE','?')}, "
            f"LR={hparams.get('LEARNING_RATE','?')}\n"
            f"```"
        )
        content = re.sub(r"```\nBest val_fid.*?```",
                         best_block, content, flags=re.DOTALL)

    PAPER_TRACKER.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run_autoresearch(n_experiments: int = 30, start_from: int = 2):
    api_key       = get_api_key()
    best_fid      = BASELINE_FID
    session_start = time.time()

    print(f"\n{'='*60}")
    print(f"  DiT-EMG Autoresearch Loop")
    print(f"  Model      : {MODEL}")
    print(f"  Experiments: {n_experiments} (from Exp {start_from})")
    print(f"  Baseline   : val_fid = {BASELINE_FID}")
    print(f"{'='*60}\n")

    for exp_n in range(start_from, start_from + n_experiments):
        elapsed_h = (time.time() - session_start) / 3600
        print(f"\n{'─'*50}")
        print(f"  EXPERIMENT {exp_n}  |  elapsed {elapsed_h:.1f}h  |  best FID {best_fid:.2f}")
        print(f"{'─'*50}")

        # ── Read context ──────────────────────────────────
        train_py = read_file(TRAIN_FILE)
        hparams  = extract_hparams_block(train_py)
        results  = read_results_summary()
        log      = read_recent_log()

        # ── Ask Claude ────────────────────────────────────
        print(f"  [claude] Thinking...")
        try:
            prompt   = build_prompt(exp_n, hparams, results, log)
            response = call_claude(prompt, api_key)
        except Exception as e:
            print(f"  [claude] Failed: {e} — skipping")
            time.sleep(30)
            continue

        # ── Parse ─────────────────────────────────────────
        parsed = parse_response(response)

        hypothesis  = parsed["hypothesis"]
        change_line = parsed["change_line"]
        new_line    = parsed["new_line"]
        log_entry   = parsed["log_entry"]

        print(f"  [claude] Hypothesis : {hypothesis[:80]}")
        print(f"  [claude] Change     : {change_line.strip()[:60]}")
        print(f"  [claude]         →  : {new_line.strip()[:60]}")

        if not change_line or not new_line:
            print("  [parse] Missing CHANGE_LINE or NEW_LINE — skipping")
            print(f"  [debug] Raw response:\n{response[:500]}")
            continue

        if change_line.strip() == new_line.strip():
            print("  [parse] Change_line equals new_line — nothing to do, skipping")
            continue

        # ── Apply change ──────────────────────────────────
        backup_train_py(exp_n)
        if not apply_line_change(change_line, new_line):
            print("  [apply] Failed to apply change — skipping")
            restore_train_py(exp_n)
            continue

        # ── Train ─────────────────────────────────────────
        metrics = run_training()

        if not metrics["success"]:
            print("  [result] Training crashed — reverting")
            restore_train_py(exp_n)
            decision = "REVERTED (crash)"
            log_experiment(exp_n, hypothesis, change_line, new_line,
                           log_entry, metrics, decision)
            continue

        # ── Evaluate ──────────────────────────────────────
        current_fid = metrics.get("fid", metrics.get("val_fid", float("inf")))

        if isinstance(current_fid, float) and current_fid < best_fid:
            pct = (best_fid - current_fid) / best_fid * 100
            print(f"  [result] ✓ IMPROVEMENT  {best_fid:.2f} → {current_fid:.2f}  (-{pct:.1f}%)")
            best_fid = current_fid
            decision = "KEEP"
        else:
            print(f"  [result] ✗ No improvement  ({current_fid} vs best {best_fid:.2f}) — reverting")
            restore_train_py(exp_n)
            decision = "REVERTED"

        # ── Log ───────────────────────────────────────────
        log_experiment(exp_n, hypothesis, change_line, new_line,
                       log_entry, metrics, decision)
        update_paper_tracker(exp_n, change_line, new_line,
                              metrics, decision, best_fid)

        print(f"\n  ── Exp {exp_n} done | decision={decision} | best FID={best_fid:.2f} ──")
        time.sleep(3)

    # ── Final summary ─────────────────────────────────────
    total_h = (time.time() - session_start) / 3600
    improvement = (BASELINE_FID - best_fid) / BASELINE_FID * 100
    print(f"\n{'='*60}")
    print(f"  Autoresearch complete!")
    print(f"  Time          : {total_h:.1f} hours")
    print(f"  Experiments   : {n_experiments}")
    print(f"  Baseline FID  : {BASELINE_FID}")
    print(f"  Best FID      : {best_fid:.4f}")
    print(f"  Improvement   : {improvement:.1f}%")
    print(f"  Logs          : {EXPERIMENT_LOG}")
    print(f"  Paper tracker : {PAPER_TRACKER}")
    print(f"  Best model    : checkpoints/best_model.pt")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=int, default=30)
    parser.add_argument("--start-from",  type=int, default=2)
    args = parser.parse_args()
    run_autoresearch(args.experiments, args.start_from)