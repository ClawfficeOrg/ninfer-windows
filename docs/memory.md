# Memory — Ornith9B on Windows 5070Ti

Date: 2026-09-18. Machine: Morrissey5070Ti (win32, RTX 5070Ti, sm_120a, CUDA 13.x).
Branch: win5070ti-port. All work uncommitted working tree.

## Status: ornith9b works

- Engine ready: ornith-1.5-9b/groupwise-int, weights 5.28 GiB.
- Smoke test: `--prompt "Hello, what model are you?" --max-new 64 --no-thinking`
  → 38 tokens, decode ~122 tok/s, prefill ~553 tok/s.
- Artifact: models/ornith_1_5_9b.ninfer (6.5 GB file, 5.28 GiB weights).

## What was ported (from ruwwww/rtx-5060ti)

Target wiring (manual, adapted to current base API):
- src/targets/qwen3_5_9b/* (config, bindings, package, variant) + registry
  + Core59/ScoreCore59 engine dispatch + full Package alias list.
- variant.h: DeviceExecutionView for gdn_norm_control_projection,
  startup_observer through create_program, core/device.h include.

Attention H16Kv4 (16,4,256), commit 71d5869a pattern:
- geometry.cuh alias, validation, split_capacity kv_heads signature,
  dispatch in prompt/small_t × BF16/FP8/K8V4/NVFP4,
  GroupSize-4 i8 warp profiles in small_t.cu.

Op shapes (checkout from fork, local untouched since merge-base
487f8977 unless noted):
- q5 linear_add plan + gemv + simt (4096 shapes).
- q4 swiglu plan (manual) + gemv kernels (checkout).
- q4 linear K64 schedule (gemv.cuh/cu + dispatch).
- attn/gdn_input q4_q5 plan + gemm + small_t + snapshot + wrappers.
- GDN gating is_9 (manual merge — local plan diverged from fork):
  kernels _9 split16/8/4/2 + unsplit, plan admit/legal/routes/execute.
- q6 dispatch (248320×4096 vocab).
- frontend chat_template digests (qwen3.5 + ornith-1.5) + tokenizer.

## Error chain hit during bring-up

1. registry undeclared Qwen3_5_9BInstance → wired.
2. create_program 4-vs-5 args → startup_observer.
3. DeviceExecutionView vs cudaStream_t → updated variant.
4. engine Core35-from-9B → Core59/ScoreCore59 + submit guard.
5. Package missing EngineCore aliases → full alias list.
6. attention unsupported head geometry → H16Kv4 port.
7. i8 static_assert PVNtPerWarp → GroupSize-4 profiles.
8. split_capacity signature → kv_heads everywhere incl. fp8/k8v4/nvfp4.
9. q5 linear_add shapes → 4096 entries.
10. BF16 GDN gating → is_9 kernels + plan.
11. q4 swiglu → plan + templated gemv.
12. Q4GemvR1W8DirectK64Schedule missing → q4 gemv header/cu/dispatch.
13. chat_template sha256 → fork digests.
14. q5 GEMV shape → fork kernels.
15. q6 linear shape → fork dispatch. Then engine ready.

## Open / not done

- Changes uncommitted. Decide: commit on win5070ti-port?
- No numerical oracle qualification run for new _9 kernels (used fork
  values verbatim). AGENTS.md wants oracle evidence for math changes.
- sampling_defaults test for ornith (fork has tests/targets/ornith_1_5_9b)
  not ported. Serving schema unaffected.
- Vision path untested (text-only smoke). MTP/speculative untested.
- 5070Ti wave-count NOTE in small_t.cu still references 5090 sweep.
