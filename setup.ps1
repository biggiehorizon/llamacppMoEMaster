# One-shot bootstrap: fetch llama.cpp sources, apply the patches if needed, and build.
#
#   git clone --recursive <this repo> llama65cpp
#   cd llama65cpp
#   .\setup.ps1
#
# Safe to re-run; it skips work that is already done.

[CmdletBinding()]
param(
    # sm_120 = Blackwell / RTX 50-series. 86 = Ampere, 89 = Ada, 90 = Hopper.
    [string] $CudaArch = '120',
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$src  = Join-Path $root 'llama.cpp'

# ---- 1. sources ----------------------------------------------------------
# llama.cpp is vendored directly into this repo (patches already applied), so a
# plain `git clone` is all that is needed. If the tree is missing, the clone was
# probably truncated by Windows' 260-char path limit -- see README.
if (-not (Test-Path (Join-Path $src 'CMakeLists.txt'))) {
    throw @"
llama.cpp\CMakeLists.txt not found.

The sources ship inside this repo, so this usually means the clone was
incomplete. llama.cpp has paths longer than 260 characters, which stock
Windows git refuses to check out. Fix with:

    git config --global core.longpaths true
    git checkout .

and clone into a short path such as C:\dev\ rather than a deeply nested one.
"@
}
Write-Host '==> sources present (vendored)' -ForegroundColor Green

# ---- 2. build ------------------------------------------------------------
$exe = Join-Path $src 'build\bin\llama-server.exe'
if ($SkipBuild) {
    Write-Host '==> skipping build (-SkipBuild)' -ForegroundColor Yellow
} elseif (Test-Path $exe) {
    Write-Host "==> llama-server.exe already built, skipping (delete $exe to force)" -ForegroundColor Green
} else {
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
        Write-Warning 'nvcc not on PATH. Install the CUDA Toolkit, or run this from a VS developer shell.'
    }
    Write-Host "==> configuring (CUDA arch $CudaArch)" -ForegroundColor Cyan
    cmake -B (Join-Path $src 'build') -S $src -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON `
        -DGGML_NATIVE=ON `
        -DCMAKE_CUDA_ARCHITECTURES=$CudaArch `
        -DLLAMA_CURL=OFF
    if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }

    Write-Host '==> building' -ForegroundColor Cyan
    cmake --build (Join-Path $src 'build') -j
    if ($LASTEXITCODE -ne 0) { throw 'build failed' }
}

# ---- 3. models -----------------------------------------------------------
# Not downloaded automatically: ~16 GB, and the right choice depends on your VRAM.
$models = Join-Path $root 'models'
if (-not (Test-Path $models)) { New-Item -ItemType Directory $models | Out-Null }

$default = Join-Path $models 'Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf'
if (-not (Test-Path $default)) {
    Write-Host ''
    Write-Host 'No model found. Download one into models\ before starting the server:' -ForegroundColor Yellow
    Write-Host '  huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \'
    Write-Host '    Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf --local-dir models'
    Write-Host ''
    Write-Host 'Use a K-quant, not an IQ-quant: CUDA 13.2 miscompiles IQ kernels for sm_120.' -ForegroundColor Yellow
} else {
    Write-Host '==> model present' -ForegroundColor Green
}

Write-Host ''
Write-Host 'Done. Start the server with:  .\start-llama-server.ps1' -ForegroundColor Green
Write-Host 'For an always-on service, see "Recreating the scheduled task" in README.md.'
