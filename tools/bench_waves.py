#!/usr/bin/env python3
"""Sequential waves with 256-token gens, one server, for hang repro."""
import subprocess
import sys
import time

sys.path.insert(0, "tools")
from bench_concurrency_long import (generate_long_prompt, run_concurrency_wave,
                                    send_chat_completion, wait_for_port)

PORT = 8097
cmd = [
    r".\build-windows\apps\Release\ninfer-serve.exe",
    "models/ornith_1_5_9b.ninfer",
    "--host", "127.0.0.1", "--port", str(PORT),
    "--max-context", "4096", "--kv-capacity", "32768",
    "--max-concurrency", "8", "--prefill-chunk", "4096",
    "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3",
    "--lm-head-draft", "--model-id", "ornith-1.5-9b",
    "--default-max-tokens", "256", "--log-level", "info",
]
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
try:
    assert wait_for_port("127.0.0.1", PORT, timeout=180.0)
    print("server ready", flush=True)
    send_chat_completion(PORT, "Hello world.", max_tokens=16)
    prompt = generate_long_prompt(target_word_count=1200)
    waves = [int(a) for a in sys.argv[1:]] or [2, 4]
    for c in waves:
        s = run_concurrency_wave(PORT, c, prompt, max_tokens=256)
        print(f"WAVE C={c}: {s['aggregate_tok_per_sec']:.2f} tok/s", flush=True)
        time.sleep(2)
finally:
    proc.terminate()
    proc.wait(timeout=15)
print("done")
