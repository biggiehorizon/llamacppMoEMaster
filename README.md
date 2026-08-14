# llama65cpp — always-on local inference rig

A personal, always-listening [llama.cpp](https://github.com/ggml-org/llama.cpp) server for
running large MoE models on a single 8 GB laptop GPU. It combines three things:

1. A **patched llama.cpp fork** that overlaps offloaded-expert weight uploads with compute,
   which is what makes large MoE models usable when most experts live in system RAM.
2. A **VRAM governor** (`start-llama-server.ps1`) that continuously re-sizes how much of the
   model lives on the GPU based on what the rest of the desktop is doing.
3. A **scheduled task** that keeps the whole thing alive across logon, crashes, and sleep.

The result is an OpenAI-compatible endpoint on the LAN that is always up, never fights games
or other GPU apps for VRAM, and drops to a ~350 MiB footprint when idle.

---

## Hardware / software baseline

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop (8151 MiB, sm_120) |
| Driver | 596.13 |
| CUDA Toolkit | 13.2 |
| OS | Windows 11 |
| Generator | Ninja, Release build |
| VRAM budget for llama | ~7.3 GB hard cap, ~5.7 GB floor |

---

## Layout

```
llama65cpp/
├─ README.md                  this file
├─ start-llama-server.ps1     watchdog + VRAM governor (the real entry point)
├─ patches/                   the three llama.cpp patches, vendored — see patches/README.md
├─ llama.cpp/                 patched fork — own git repo, git-ignored, see "The fork" below
│  └─ build/bin/llama-server.exe
└─ models/                    GGUF weights + logs (git-ignored, ~30 GB)
   ├─ Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf        15.7 GB  ← currently served
   ├─ gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf 13.3 GB
   ├─ server.log              llama-server's own log
   └─ governor.log            every scale-up / scale-down decision
```

---

## The fork

The `llama.cpp/` working copy is **not** tracked by this repo — it is its own git clone. What
*is* tracked is `patches/`, the three commits exported with `git format-patch`, so a working
build can be reproduced from this repo plus an upstream clone. See
[`patches/README.md`](patches/README.md) for the base commit and the `git am` invocation.

`llama.cpp/` tracks `github.com/thecodacus/llama.cpp`, branch **`fable5/prefetch-experts`**,
which carries three commits on top of upstream master. All three target the same bottleneck:
when experts are offloaded to CPU (`--n-cpu-moe`), prefill is dominated by host→device copies
of expert weights, and by default those copies block compute.

### 1. `llama : pin mmap-backed CPU weights for faster H2D uploads`

After the model finishes loading, the mmap pages backing weights that stayed in system memory
are registered with the CUDA backend (`cudaHostRegister`), turning pageable copies into pinned
ones. Opt-in via `GGML_CUDA_REGISTER_HOST=1`; behaviour is unchanged without it.

Measured on an RTX 3060: Qwen3.6-35B-A3B `pp2048` **1144 → 1385 t/s**.

> ⚠️ **Windows caveat:** `llama_mmap::register_host()` is guarded by `#ifdef _POSIX_MAPPED_FILES`,
> so on this rig it compiles to a no-op returning 0 and you will never see the
> `pinned N MiB of mapped model memory` log line. The speedup from this commit is currently
> Linux/macOS only. Porting it would mean adding a `_WIN32` branch that page-aligns using
> `GetSystemInfo().dwPageSize` instead of `sysconf(_SC_PAGESIZE)`.

### 2. `ggml : overlap offloaded expert weight uploads with compute`

The core change, in `ggml_backend_sched_compute_splits()`. Normally the scheduler waits for the
routing ids before copying expert weights in. The insight: **with a large batch, virtually every
expert gets used anyway**, so waiting for routing buys nothing. Instead the full expert tensor is
uploaded eagerly through a *second backend instance on the same device*, so the copy overlaps the
previous split's compute, with CUDA events synchronising a ring of staging slots.

Guards, all of which must hold or it falls back to the stock path:

- device reports `caps.async` **and** `caps.events`
- the split's first node is `GGML_OP_MUL_MAT_ID` whose `src[0]` is the copied input
- the input buffer is host memory tagged `USAGE_WEIGHTS`
- the batch is big enough to be worth it: `ids->ne[0]*ids->ne[1] >= 2 * n_expert`
- no `callback_eval` set (i.e. not running under a debug/eval callback)

### 3. `ggml : size prefetch slots per layer and fix fallback use-after-free`

Generalises the fixed 2 slots into a configurable ring (max 8), sized **once per graph** from the
largest offloaded expert tensor so slots never have to grow mid-eval, and fixes a use-after-free
when prefetch is disabled partway through (buffers are now freed only after both backends
synchronise).

### Tuning knob

```
GGML_SCHED_PREFETCH_EXPERTS=0   off (default)
GGML_SCHED_PREFETCH_EXPERTS=1   on, default 3 slots — covers gate/up/down of one MoE layer
GGML_SCHED_PREFETCH_EXPERTS=N   on, N slots (clamped to GGML_SCHED_MAX_PREFETCH_SLOTS = 8)
```

More slots let uploads run further ahead of compute; each costs one max-sized expert tensor of
VRAM. It also requires `op_offload` to be enabled — with it off, prefetch is silently inactive.

### Keeping up with upstream

```powershell
cd C:\dev\llama65cpp\llama.cpp
git fetch origin
git rebase origin/master          # the three commits are small and touch narrow surfaces
```

Rebase conflicts, if any, will land in `ggml/src/ggml-backend.cpp` around
`ggml_backend_sched_compute_splits`, which is a hot file upstream.

---

## Building

```powershell
cd C:\dev\llama65cpp\llama.cpp
cmake -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DGGML_NATIVE=ON `
  -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DLLAMA_CURL=OFF
cmake --build build -j
```

`CMAKE_CUDA_ARCHITECTURES=120` is sm_120 (Blackwell / RTX 50-series). Building only the one arch
keeps compile times sane. `LLAMA_CURL=OFF` avoids the libcurl dependency since models are
downloaded manually.

Output lands at `llama.cpp\build\bin\llama-server.exe`.

---

## Running

The server is **not** started by hand — `start-llama-server.ps1` owns its lifecycle. But the
command it builds is:

```powershell
$env:GGML_SCHED_PREFETCH_EXPERTS = '1'
.\llama-server.exe `
  -m ..\..\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf `
  -ngl 99 --n-cpu-moe <40 - gpuLayers> `
  -c 65536 --parallel 2 `
  -b 2048 -ub 2048 `
  --host 0.0.0.0 --port 9090 `
  --sleep-idle-seconds 900 `
  --log-file ..\..\models\server.log
```

Notes on the choices:

- `-ngl 99` puts *everything* on the GPU, then `--n-cpu-moe N` claws the expert weights of `N`
  layers back to the CPU. This is the lever the governor actually turns.
- `-c 65536` with `--parallel 2` gives two concurrent 32K-context slots.
- `-b 2048 -ub 2048` — a large physical batch is what makes the expert prefetch pay off
  (see the `>= 2 * n_expert` guard above). Shrinking `-ub` disables the optimisation in practice.
- `--sleep-idle-seconds 900` releases VRAM after 15 idle minutes, dropping to ~350 MiB.

### Endpoint

```
http://<your-host>:9090        (LAN)
http://127.0.0.1:9090               (local)
```

OpenAI-compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
Health and state: `/health`, `/props` (`is_sleeping`).

> ⚠️ **Never poll `/slots`.** It resets the server's 15-minute sleep timer, so the model never
> unloads and the governor loses its only safe window to scale *up*. The governor deliberately
> infers busy/idle from the `launch_slot` / `slot release` lines in `server.log` instead.

---

## The VRAM governor

`start-llama-server.ps1` runs at logon and then **every 3 minutes** via the `Llama Server`
scheduled task:

```
conhost.exe --headless powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "C:\dev\llama65cpp\start-llama-server.ps1"
```

Each tick it does three things:

**1. Watchdog.** `GET /health`. If unhealthy: a process younger than 5 minutes is assumed to
still be loading and left alone; anything older is killed and restarted.

**2. Accounting.** Reads `nvidia-smi` for used/total VRAM, reads the live `--n-cpu-moe` value
straight off the running process's command line (`Win32_Process.CommandLine`), and subtracts its
own estimated footprint to derive how much VRAM *other apps* are holding:

```
ours   = sleeping ? 350 MiB : 4290 + layers * 340 MiB
others = used - ours
```

| GPU expert layers | Estimated VRAM |
|---|---|
| 4 (`MIN_L`, floor) | ~5650 MiB |
| 6 | ~6330 MiB |
| 8 (typical settled value) | ~7010 MiB |
| 9 (`MAX_L`, cap) | ~7350 MiB |

`BASE_MIB = 4290` covers non-expert weights + 2×32K context + the `ub 2048` compute buffers +
the prefetch slots. `PER_LAYER = 340` is expert weights per layer, for a 40-layer model.

**3. Decision.** Two asymmetric thresholds:

- **Scale down** (`DOWN_RESERVE = 300`) when free VRAM would fall below 300 MiB — but only if
  the server is idle. If a request is in flight, it logs the intent and retries next tick.
- **Scale up** (`UP_RESERVE = 800`) only requires 800 MiB of headroom to remain, and **only
  happens while the server is asleep**, so growing never interrupts anyone.

There is also a WDDM correction: under GPU overcommit Windows demotes other apps' pages out of
VRAM, which makes them invisible to the `others` estimate. So if absolute free VRAM drops below
500 MiB, the governor yields a layer regardless of what the arithmetic says.

Scaling either direction means killing and relaunching the server — llama.cpp can't re-shard a
loaded model — so scale events cost a model reload. That's why the thresholds are wide.

Every decision is appended to `models\governor.log`, which self-truncates to the last 300 lines
once it passes 200 KB.

> 📌 The header comment in the script says the baseline is 6 layers, but `$MIN_L` is actually `4`.
> The code is the source of truth; the comment is stale.

### Recreating the scheduled task

```powershell
$action  = New-ScheduledTaskAction -Execute 'conhost.exe' `
  -Argument '--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\llama65cpp\start-llama-server.ps1"'
$logon   = New-ScheduledTaskTrigger -AtLogOn
$repeat  = New-ScheduledTaskTrigger -Once -At (Get-Date) `
  -RepetitionInterval (New-TimeSpan -Minutes 3)
Register-ScheduledTask -TaskName 'Llama Server' -Action $action -Trigger $logon,$repeat
```

`conhost.exe --headless` is what keeps a console window from flashing every 3 minutes.

---

## Models

| Model | Quant | Size | Layers | Notes |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | UD-Q3_K_XL | 15.7 GB | 40 | current default |
| Gemma 4 26B A4B (it-qat) | UD-Q4_K_XL | 13.3 GB | — | previous default |

Both are Unsloth Dynamic quants. Weights are **not** in this repo — see `.gitignore`.

> ⚠️ **K-quants only on this rig.** CUDA 13.2 miscompiles the IQ-quant kernels for sm_120 and
> produces gibberish on batched prefill. (Reported in the Unsloth Qwen3.6 HF discussion #3.)
> If you switch to an `IQ*` quant and output turns to noise at large `-b`, this is why.

Switching models means updating `$model`, `$N_LAYERS`, `$BASE_MIB`, and `$PER_LAYER` at the top
of `start-llama-server.ps1` — the governor's arithmetic is model-specific.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| Endpoint down | `Get-Content models\governor.log -Tail 20` — look for repeated kills |
| Restart loop | `models\server.log` tail; usually OOM at the current layer count |
| Gibberish output at high batch | IQ-quant on CUDA 13.2 — switch to a K-quant |
| Never scales up | Something is polling `/slots` and resetting the sleep timer |
| Stuck at 4 layers | Another GPU app is holding VRAM; check `nvidia-smi` |
| No `pinned N MiB` log line | Expected on Windows — that path is POSIX-only |

---

## Licensing / attribution

`llama.cpp` is MIT-licensed, so maintaining a private or public fork is fine as long as the
upstream `LICENSE` and copyright notice stay intact — they do. The three patches above are
authored by `thecodacus <thecodacus@gmail.com>`; preserve that authorship if this is ever
published or upstreamed.

This repo (the governor script and these notes) is separate from the fork and carries no
upstream code.
