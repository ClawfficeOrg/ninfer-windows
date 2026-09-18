#!/usr/bin/env python3
"""Run sequential waves against one server, optional prefix reuse toggle."""
import subprocess
import sys
import time

sys.path.insert(0, "tools")
from bench_concurrency_long import (generate_long_prompt, run_concurrency_wave,
                                    send_chat_completion, wait_for_port)

PORT = 8096
LOG = r"C:\Users\hippo\AppData\Local\Temp\opencode\serve_reuse.log"

cmd = [
    r".\build-windows\apps\Release\ninfer-serve.exe",
    "models/ornith_1_5_9b.ninfer",
    "--host", "127.0.0.1", "--port", str(PORT),
    "--max-context", "4096", "--kv-capacity", "32768",
    "--max-concurrency", "8", "--prefill-chunk", "4096",
    "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3",
    "--lm-head-draft", "--model-id", "ornith-1.5-9b",
    "--default-max-tokens", "256", "--log-level", "debug",
]
if "--no-reuse" in sys.argv:
    cmd.append("--no-prefix-reuse")

logf = open(LOG, "w")
proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
try:
    assert wait_for_port("127.0.0.1", PORT, timeout=180.0), "start timeout"
    print("server ready", flush=True)
    send_chat_completion(PORT, "Hello world.", max_tokens=16)
    prompt = generate_long_prompt(target_word_count=1200)
    for c in (2, 2):
        s = run_concurrency_wave(PORT, c, prompt, max_tokens=64)
        print(f"WAVE C={c}: {s['aggregate_tok_per_sec']:.2f} tok/s", flush=True)
        time.sleep(2)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    logf.close()
print("--- log tail ---")
with open(LOG) as f:
    print("".join(f.readlines()[-12:]))
