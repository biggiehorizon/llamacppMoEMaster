# Llama server watchdog + VRAM governor. Run by the "Llama Server" scheduled
# task at logon and every few minutes.
#
# Keeps the server always listening on :9090 and dynamically sizes how many MoE
# expert layers live in VRAM based on what other applications are using:
#   baseline 6 GPU layers (~5.8 GB total) ... max 9 GPU layers (~7.1 GB, under
#   the 7.2 GB cap), floor stays at baseline no matter the pressure.
# Scale-down happens when free VRAM is tight (applied while idle or sleeping);
# scale-up only while the server sleeps, so it never interrupts anything.
# Decisions are appended to governor.log next to the model.

# Paths are derived from this script's location, so the repo works wherever it
# is cloned. $PSScriptRoot resolves correctly under the scheduled task too,
# because the task invokes powershell.exe with -File and an absolute path.
$root   = $PSScriptRoot
$exe    = Join-Path $root 'llama.cpp\build\bin\llama-server.exe'
$model  = Join-Path $root 'models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf'
$logDir = Join-Path $root 'models'
$srvLog = "$logDir\server.log"
$govLog = "$logDir\governor.log"

# NOTE: K-quants only for this rig - CUDA 13.2 miscompiles IQ-quant kernels
# (gibberish on batch prefill, see huggingface unsloth Qwen3.6 discussion #3)
$N_LAYERS     = 40    # MoE layers in the model
$BASE_MIB     = 4290  # non-expert weights + 2x32K ctx + ub2048 compute buffers + prefetch slots
$PER_LAYER    = 340   # MiB of expert weights per layer
$SLEEP_MIB    = 350   # our footprint while sleeping
$MIN_L        = 4     # baseline (~5.7 GB)
$MAX_L        = 9     # cap (~7.3 GB; UP_RESERVE usually settles it at 8)
$UP_RESERVE   = 800   # only grow if this much stays free afterwards
$DOWN_RESERVE = 300   # shrink when free VRAM would drop below this

function Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg" | Add-Content $govLog
    if ((Get-Item $govLog -ErrorAction SilentlyContinue).Length -gt 200KB) {
        Get-Content $govLog -Tail 300 | Set-Content $govLog -Encoding utf8
    }
}

function Start-Server($gpuLayers) {
    $env:GGML_SCHED_PREFETCH_EXPERTS = '1'
    Start-Process -FilePath $exe -WindowStyle Hidden -ArgumentList @(
        '-m', "`"$model`"",
        '-ngl', '99', '--n-cpu-moe', "$($N_LAYERS - $gpuLayers)",
        '-c', '65536', '--parallel', '2',
        '-b', '2048', '-ub', '2048',
        '--host', '0.0.0.0', '--port', '9090',
        '--sleep-idle-seconds', '900',
        '--log-file', "`"$srvLog`""
    )
    Log "started server with $gpuLayers expert layers on GPU (~$($BASE_MIB + $gpuLayers*$PER_LAYER) MiB)"
}

# ---- GPU state ----
$smi = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -split ','
$usedMiB  = [int]$smi[0].Trim()
$totalMiB = [int]$smi[1].Trim()

function Target-Layers($othersMiB, $reserve) {
    $budget = [math]::Min(7373, $totalMiB - $othersMiB - $reserve)
    $fit = [math]::Floor(($budget - $BASE_MIB) / $PER_LAYER)
    return [int][math]::Max($MIN_L, [math]::Min($MAX_L, $fit))
}

# ---- server state ----
$healthy = $false
try { Invoke-RestMethod 'http://127.0.0.1:9090/health' -TimeoutSec 5 | Out-Null; $healthy = $true } catch {}

if (-not $healthy) {
    # stuck process still loading? give it time; hung for good? kill it
    $p = Get-Process llama-server -ErrorAction SilentlyContinue
    if ($p) {
        if (((Get-Date) - $p.StartTime).TotalMinutes -lt 5) { exit 0 }
        Stop-Process -Id $p.Id -Force -Confirm:$false
        Start-Sleep -Seconds 3
        Log "killed unresponsive server (pid $($p.Id))"
    }
    # everything currently in VRAM belongs to others
    Start-Server (Target-Layers $usedMiB $UP_RESERVE)
    exit 0
}

# current config from the live process command line
$cmd = (Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'").CommandLine
if ($cmd -notmatch '--n-cpu-moe (\d+)') { exit 0 }
$layersNow = $N_LAYERS - [int]$Matches[1]

$sleeping = $false
try { $sleeping = [bool](Invoke-RestMethod 'http://127.0.0.1:9090/props' -TimeoutSec 5).is_sleeping } catch {}

$oursMiB   = if ($sleeping) { $SLEEP_MIB } else { $BASE_MIB + $layersNow * $PER_LAYER }
$othersMiB = [math]::Max(0, $usedMiB - $oursMiB)

$targetDown = Target-Layers $othersMiB $DOWN_RESERVE
$targetUp   = Target-Layers $othersMiB $UP_RESERVE

# busy = launch_slot logged after the last release (covers long prefills that
# go quiet in the log); ignore stale logs so a mid-generation crash long ago
# can't wedge the governor. NOTE: never poll /slots here - it resets the
# server's 15-min sleep timer.
$busy = $false
if (-not $sleeping -and (Test-Path $srvLog)) {
    $logAgeMin = ((Get-Date) - (Get-Item $srvLog).LastWriteTime).TotalMinutes
    if ($logAgeMin -lt 30) {
        $tail = Get-Content $srvLog -Tail 80
        $lastLaunch = -1; $lastRelease = -1
        for ($i = 0; $i -lt $tail.Count; $i++) {
            if ($tail[$i] -match 'launch_slot') { $lastLaunch = $i }
            if ($tail[$i] -match 'slot\s+release') { $lastRelease = $i }
        }
        $busy = $lastLaunch -gt $lastRelease
    }
}

# WDDM demotes other apps' pages under overcommit, hiding them from the
# "others" estimate - low absolute free VRAM means we should yield a layer
$freeMiB = $totalMiB - $usedMiB
if (-not $sleeping -and $freeMiB -lt 500 -and $targetDown -ge $layersNow -and $layersNow -gt $MIN_L) {
    $targetDown = $layersNow - 1
}

if ($layersNow -gt $targetDown) {
    if ($busy) { Log "want $targetDown layers (down from $layersNow, others=$othersMiB MiB) but server is busy; retry next tick"; exit 0 }
    Log "scaling DOWN $layersNow -> $targetDown layers (others=$othersMiB MiB, used=$usedMiB/$totalMiB)"
    Stop-Process -Name llama-server -Force -Confirm:$false
    Start-Sleep -Seconds 3
    Start-Server $targetDown
} elseif ($sleeping -and $targetUp -gt $layersNow) {
    Log "scaling UP $layersNow -> $targetUp layers while sleeping (others=$othersMiB MiB, used=$usedMiB/$totalMiB)"
    Stop-Process -Name llama-server -Force -Confirm:$false
    Start-Sleep -Seconds 3
    Start-Server $targetUp
}
