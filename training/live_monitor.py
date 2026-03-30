#!/usr/bin/env python3
"""Background monitor: exports training stats and live replays to web/ periodically."""

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
WEB_DIR = TRAINING_DIR.parent / "web"
MODELS_DIR = TRAINING_DIR / "models"
REFRESH_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "180"))  # seconds


def find_latest_log():
    logs = sorted(glob.glob(str(TRAINING_DIR / "logs/*.log")), key=os.path.getmtime)
    return logs[-1] if logs else None


def parse_log(log_path):
    """Extract training history from log file."""
    history = []
    stage = 1
    step_pat = re.compile(
        r"step=\s*(\d+)\s*\|\s*eps/s=\s*([\d.]+)\s*\|\s*success=\s*([\d.]+)%.*elapsed=\s*([\d.]+)s"
    )
    with open(log_path) as f:
        for line in f:
            if "Stage 3" in line:
                stage = 3
            elif "Stage 2" in line:
                stage = 2
            m = step_pat.search(line)
            if m:
                history.append({
                    "step":    int(m.group(1)),
                    "eps_s":   float(m.group(2)),
                    "success": float(m.group(3)),
                    "elapsed": float(m.group(4)),
                    "stage":   stage,
                })
    return history, stage


def find_latest_checkpoint(stage):
    pattern = str(MODELS_DIR / f"ppo_stage{stage}_checkpoints/ckpt_*_steps.zip")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
    return checkpoints[-1] if checkpoints else None


def export_replays(checkpoint_path, output_path, difficulty):
    result = subprocess.run(
        [sys.executable, str(TRAINING_DIR / "export_replays.py"),
         checkpoint_path, "-o", str(output_path), "-n", "30", "-d", str(difficulty)],
        capture_output=True, text=True, cwd=str(TRAINING_DIR),
    )
    return result.returncode == 0, result.stdout + result.stderr


def run():
    print(f"[live_monitor] started — interval={REFRESH_INTERVAL}s, web={WEB_DIR}", flush=True)
    WEB_DIR.mkdir(exist_ok=True)

    while True:
        try:
            log_path = find_latest_log()
            if log_path:
                history, current_stage = parse_log(log_path)

                if history:
                    last = history[-1]
                    stats = {
                        "stage":       current_stage,
                        "step":        last["step"],
                        "eps_s":       last["eps_s"],
                        "success":     last["success"],
                        "elapsed":     last["elapsed"],
                        "log_file":    os.path.basename(log_path),
                        "last_update": time.strftime("%H:%M:%S"),
                        "history":     history[-60:],  # last 60 data points for chart
                    }
                    with open(WEB_DIR / "training_live.json", "w") as f:
                        json.dump(stats, f, separators=(",", ":"))
                    print(
                        f"[live_monitor] stage={current_stage} "
                        f"step={last['step']:,} success={last['success']:.1f}%",
                        flush=True,
                    )

                ckpt = find_latest_checkpoint(current_stage)
                if ckpt:
                    ok, msg = export_replays(ckpt, WEB_DIR / "replays_live.json", current_stage)
                    ckpt_name = os.path.basename(ckpt)
                    if ok:
                        print(f"[live_monitor] replays exported from {ckpt_name}", flush=True)
                    else:
                        print(f"[live_monitor] replay export failed: {msg[:120]}", flush=True)
        except Exception as e:
            print(f"[live_monitor] error: {e}", flush=True)

        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    run()
