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

> ### Credit
>
> The three llama.cpp optimizations at the heart of this setup — the expert-upload/compute
> overlap, the prefetch slot ring, and the pinned-host-memory path — are the work of
> **[Codacus](https://github.com/thecodacus)** (`thecodacus`), from the
> [`fable5/prefetch-experts`](https://github.com/thecodacus/llama.cpp/tree/fable5/prefetch-experts)
> branch of their llama.cpp fork. They are what make a 35B MoE model practical on 8 GB of VRAM;
> without them this repo would just be a launch script.
>
> Everything original here is the packaging around that work: the VRAM governor, the bootstrap
> script, and these notes. Full attribution is preserved in the git authorship of every commit
> in [`patches/`](patches/).
>
> llama.cpp itself is by [Georgi Gerganov and the ggml authors](https://github.com/ggml-org/llama.cpp).

---

## Quick start

Requires the CUDA Toolkit (13.x), CMake, Ninja, and an NVIDIA GPU. Run from a
Visual Studio developer shell so the host compiler is on `PATH`.

```powershell
git config --global core.longpaths true    # required on Windows, see note below
git clone https://github.com/<you>/llama65cpp C:\dev\llama65cpp
cd C:\dev\llama65cpp
.\setup.ps1                                # add -CudaArch 89 for Ada, 86 for Ampere, etc.
```

The patched llama.cpp sources ship **inside this repo** — there is no submodule and nothing
else to fetch. `setup.ps1` configures and builds `llama-server.exe`, then tells you which model
to download. Grab one (~16 GB):

```powershell
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF `
  Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf --local-dir models
```

And start it:

```powershell
.\start-llama-server.ps1
```

The server comes up on `http://0.0.0.0:9090` with an OpenAI-compatible API. For an always-on
setup, register the scheduled task — see [The VRAM governor](#the-vram-governor) below.

> ⚠️ **Windows long paths — set this before cloning.** llama.cpp has source paths longer than
> 260 characters (under `tools/ui/`), so stock Windows git aborts the checkout with
> `Filename too long`, leaving a partial tree. Run `git config --global core.longpaths true`
> first, and clone into a short path like `C:\dev\` rather than somewhere deeply nested. If you
> already cloned and files are missing, enable the setting and run `git checkout .` to recover.

**Adapting to other hardware:** the governor's VRAM arithmetic is tuned for an 8 GB card and
this specific model. On different hardware, adjust `$MIN_L`, `$MAX_L`, `$BASE_MIB`, and
`$PER_LAYER` at the top of `start-llama-server.ps1` — see
[The VRAM governor](#the-vram-governor) for what each means.

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
├─ LICENSE                    MIT (this repo); llama.cpp keeps its own at llama.cpp/LICENSE
├─ setup.ps1                  one-shot bootstrap: sources → build → model instructions
├─ start-llama-server.ps1     watchdog + VRAM governor (the real entry point)
├─ patches/                   the three patches as standalone files, for upstreaming/rebasing
├─ llama.cpp/                 full patched llama.cpp source, tracked in this repo
│  ├─ models/                 upstream tokenizer test fixtures (tracked, ~75 MB)
│  └─ build/bin/llama-server.exe   (build output, git-ignored)
└─ models/                    GGUF weights + logs (git-ignored, ~30 GB)
   ├─ Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf        15.7 GB  ← currently served
   ├─ gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf 13.3 GB
   ├─ server.log              llama-server's own log
   └─ governor.log            every scale-up / scale-down decision
```

---

## The fork

`llama.cpp/` holds the **full upstream source with the patches already applied**, tracked
directly in this repo. Nothing is fetched at build time and there is no dependency on any
external fork staying available — clone and build.

The sources were vendored from [Codacus's llama.cpp fork](https://github.com/thecodacus/llama.cpp),
branch `fable5/prefetch-experts`, at commit `5f83fbb` — which is upstream
[`4fc4ec554`](https://github.com/ggml-org/llama.cpp/commit/4fc4ec5541b243957ae5099edb67372f8f3b550e)
plus three commits. `patches/` keeps those three as standalone `git format-patch` files, which
is what you want for rebasing onto newer upstream or upstreaming them.

All three target the same bottleneck:
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

Because the sources are vendored, `llama.cpp/` has no git history of its own — this repo's
history *is* its history. To move to a newer upstream, rebuild the tree in a scratch clone and
copy the result back:

```powershell
# 1. rebase the three patches onto current upstream, in a scratch clone
git clone https://github.com/ggml-org/llama.cpp $env:TEMP\lcpp
cd $env:TEMP\lcpp
git checkout -b prefetch-experts
git am C:\dev\llama65cpp\patches\*.patch      # resolve conflicts here if any

# 2. re-export the patches and refresh the vendored tree
git format-patch -3 --no-numbered --zero-commit -o C:\dev\llama65cpp\patches
Remove-Item -Recurse -Force .git
robocopy . C:\dev\llama65cpp\llama.cpp /MIR /XD build

# 3. rebuild, verify, then commit the update in one go
cd C:\dev\llama65cpp; .\setup.ps1
```

Then update the base commit hash in [`patches/README.md`](patches/README.md).

Conflicts, if any, will land in `ggml/src/ggml-backend.cpp` around
`ggml_backend_sched_compute_splits()`, which is a hot file upstream.

> `robocopy /MIR` mirrors deletions too, which is what you want — but it is destructive.
> Make sure your working tree is committed first so `git diff` can show you exactly what moved.

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
http://<your-host>:9090             (LAN)
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

This repository is **MIT-licensed** — see [`LICENSE`](LICENSE). That grant covers the original
work here: `start-llama-server.ps1`, `setup.ps1`, this README, and `patches/`.

It also **contains a full copy of llama.cpp**, which is independently MIT-licensed.
Redistribution is explicitly permitted provided the copyright notice and license text travel
with it — they do, verbatim, at [`llama.cpp/LICENSE`](llama.cpp/LICENSE). llama.cpp bundles
third-party code under its own terms in `llama.cpp/vendor/` and `llama.cpp/licenses/`; those are
intact too. Both licenses being MIT means there is no compatibility question.

The three patches in `patches/` are authored by **Codacus**
(`thecodacus <thecodacus@gmail.com>`) — see [Credit](#credit). `git am` preserves that
authorship, and it must be preserved if they are ever submitted upstream. This repo
redistributes them, it does not claim them.

> Note: llama.cpp does **not** accept predominantly AI-generated pull requests, and asks that
> contributors fully understand and be able to maintain any code they submit. See
> `llama.cpp/AGENTS.md` before considering upstreaming anything.

---

## How the sources are vendored

Notes for anyone maintaining or re-doing the vendoring, and a record of what was verified.

**What is tracked:** all 3,019 files of the llama.cpp tree, matching upstream exactly.
`.git` is ~48 MB. Model weights are excluded; the largest tracked file is a 15 MB tokenizer
fixture, comfortably under GitHub's 100 MB per-file limit.

**Fidelity check.** The vendored tree hashes to `1cbce4cc79442bc287c823937d965d7b73b2e0c6`,
which is byte-for-byte identical to fork commit `5f83fbb`, file modes included. To re-verify
after any future re-vendoring:

```powershell
git rev-parse HEAD:llama.cpp        # must equal the source commit's tree hash
```

Copying a git working tree into another repo is lossier than it looks. Three things bit us,
all of which will bite again on a re-vendor:

1. **Ignore rules match at any depth.** A bare `models/` in `.gitignore` also matches
   `llama.cpp/models/`, silently dropping 111 tokenizer fixtures the test suite needs. The rules
   are anchored with a leading slash (`/models/`) for exactly this reason — don't un-anchor them.
2. **Upstream force-adds three files past its own `.gitignore`** (`build-xcframework.sh`, a
   benchmark `.log`, and an Xcode plist). A plain `git add` drops them; they need `git add -f`.
3. **Windows does not preserve file modes.** Vendoring from a Windows checkout flattens every
   `100755` file to `100644`, so 117 scripts — `ci/run.sh`, the `convert_*.py` tools — arrive
   non-executable for anyone on Linux or macOS. Restore with `git update-index --chmod=+x`.

The reliable way to catch all three is to diff against the source repo rather than eyeball it:

```powershell
# file list, contents, and modes, in one comparison
git --git-dir=<source>\.git ls-tree -r <commit> | Sort-Object > up.txt
git ls-tree -r HEAD:llama.cpp        | Sort-Object > ours.txt
Compare-Object (Get-Content up.txt) (Get-Content ours.txt)
```

**`.fork-git`.** The 420 MB git history of the fork these sources came from is kept aside at
`.fork-git` (git-ignored) so the original commits remain available. Restore it with
`mv .fork-git llama.cpp\.git`, or delete it — the patches in `patches/` carry the same changes.
