# Offline Python Coding Assistant on the Snapdragon X Elite NPU

Entry for the Snapdragon AI Lab Build & Present Challenge. A small on-device LLM that acts as an offline Python coding tutor, running on the Snapdragon X Elite NPU.

**Testing note:** I do not own a Snapdragon laptop. All on-device runs were done remotely on **Qualcomm Device Cloud** (device CRD8380X, SC8380XP, Windows 11).

## Stack
- Model: `qualcomm/Qwen3-1.7B:w4a16` (official Qualcomm bundle, 1.6 GiB)
- Runtime: GenieX CLI v0.7.0 (QAIRT 2.45 bundled)
- Load line reported: `runtime=qairt compute=npu`

## Reproduce (on a Windows on Snapdragon device)
```powershell
# 1. Install GenieX CLI (silent)
geniex-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART

# 2. Pull the model from AI Hub (about 1.6 GiB)
geniex pull qualcomm/Qwen3-1.7B:w4a16 --model-hub aihub --model-type llm --skip-update

# 3. Run on the NPU
geniex infer qualcomm/Qwen3-1.7B:w4a16 --think=false --skip-update --verbose -s "You are an expert coding assistant. Give concise, correct code."
```
If `geniex` is not on PATH, call it by full path (the install folder contains a space, so quote it).

## Measured results (single short runs, indicative only)
| Metric | Value |
|---|---|
| Decode speed | ~40-42 tok/s typical (range 39-45) |
| Time to first token | ~0.05 s (0.052-0.058 s; one 0.7 s outlier) |
| Prompt processing | 364-733 tok/s |

Demo prompts with correct answers: reverse a string, binary search, Fibonacci generator.

## Honest notes
- First attempt, Qwen2.5-Coder-1.5B exported to ONNX, failed at NPU profiling (the NPU path needs w4a16), so I switched to the prebuilt Qwen3-1.7B w4a16.
- Qwen3-1.7B is a small general model, not code-specialised. One decorator prompt gave a wrong answer in an earlier take.
- Numbers are from a few short runs, not a formal benchmark.

## Repo contents
- `benchmark_snapdragon.py`: earlier Qualcomm AI Hub job experiment. Reads `QAI_HUB_API_TOKEN` from the environment; no secrets are stored in this repo.
- `run_tough_benchmark.py`: GPU-side prompt set used to compare outputs on a workstation (not the on-device measurement).
- `docs/`: pitch deck (PDF and PPTX) and project description.

## Next steps
Code-tuned w4a16 model, proper correctness and latency benchmark, local chat UI, offline classroom installer.
