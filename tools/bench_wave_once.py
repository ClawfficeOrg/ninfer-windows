#!/usr/bin/env python3
"""Run one concurrency wave against a freshly started serve instance, keep server logs."""
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "tools")
from bench_concurrency_long import generate_long_prompt, send_chat_completion, wait_for_port

PORT = 8094
LOG = r"C:\Users\hippo\AppData\Local\Temp\opencode\serve_c4.log"

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

logf = open(LOG, "w")
proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
try:
    if not wait_for_port("127.0.0.1", PORT, timeout=180.0):
        print("SERVER START TIMEOUT"); sys.exit(1)
    print("server ready")
    send_chat_completion(PORT, "Hello world.", max_tokens=16)
    print("warmup done")
    prompt = generate_long_prompt(target_word_count=1200)
    conc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    prompts = [prompt + f"\n[Session Lane Index: {i+1}]" for i in range(conc)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(send_chat_completion, PORT, p, 256) for p in prompts]
        results = [f.result(timeout=280.0) for f in futs]
    dt = time.perf_counter() - t0
    tot = sum(r["completion_tokens"] for r in results)
    print(f"C={conc}: {tot} tok in {dt:.2f}s = {tot/dt:.2f} tok/s")
    for i, r in enumerate(results):
        print(f"  lane {i+1}: {r['completion_tokens']} tok {r['elapsed']:.2f}s")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    logf.close()
print("--- server log tail ---")
with open(LOG) as f:
    lines = f.readlines()
print("".join(lines[-30:]))
