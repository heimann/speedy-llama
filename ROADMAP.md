# Roadmap

Priorities in order: compatibility, then speed, then ergonomics.

## Now

- **cohere2_moe** (North-Mini-Code-1.0) - done. [Quants on HF](https://huggingface.co/heimann/North-Mini-Code-1.0-GGUF), imatrix-calibrated Q4_K_M, correct chat template embedded.
- Track upstream: keep the fork delta small (arch patches, presets, docs only). Automate a daily upstream sync.

## Next

- **Speculative decoding presets** - llama.cpp supports it; nobody ships defaults. Per supported model: known-good draft pairing, draft counts, p-min. One command, not forty flags.
- **Dual-GPU speculative decoding** - target model on the big card, draft on the small one (`-dev`/`-devd`). Tuned presets and honest benchmarks for common consumer pairs (3090+3080, 4090+anything).
- **Opinionated guides** - exact commands for specific cards, no flag tours:
  - serving fast quants remotely over Tailscale
  - hooking up Open WebUI
  - routing different applications to different models
  - (open question: ship an Ollama-compatible API shim? many clients speak it)

## Later

- **MTP self-speculation** - upstream merged MTP speculative decoding in May 2026 (PR #22673, `--spec-type draft-mtp`; Qwen3.6 GGUFs ship MTP heads, ~1.5-2x claimed). The work here is now: tuned MTP presets + honest 4090/3090 benchmarks for supported models, and wiring the North MTP variant (cohere2_moe NextN graph, per PR #24260) into the merged machinery - a combination only this fork can run.
- **Daemon** (separate repo, BEAM/OTP): Ollama ergonomics on llama.cpp performance.
  - idle unload after TTL
  - hot model swap on the request's `model` field (queue requests during swaps)
  - VRAM-aware: picks quant + context that fit the card, refuses combos that OOM
  - second-card-aware: automatically places a draft model on a smaller GPU for speculative decoding when present
  - one command to serve a model on your tailnet
- **Throughput presets** - continuous batching configs per card tier for multi-user and batch workloads.

## Infrastructure notes

- Cloud jobs (imatrix, bf16 evals): keep a persistent RunPod network volume with converted GGUFs - pay the 60GB-download + conversion tax once per model.
- During cloud jobs, do heavy sequential I/O on container-local disk, not the network-mounted `/workspace` (MooseFS; killed a 57GB conversion mid-write).
- Smoke tests for new archs must include multi-turn conversational prompts, not just one-shot code tasks - the wrong-chat-template bug only manifested in chat.
