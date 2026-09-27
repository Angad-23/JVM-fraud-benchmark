# run_all.ps1 - runs every latency arm in a fixed order, unattended.
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File <project>\scripts\run_all.ps1
# Prerequisites: laptop on charger, Best performance mode, PowerThrottlingOff=1 + reboot done,
#                apps closed, OneDrive paused. Do NOT touch the laptop while it runs (~20-25 min).

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$src  = Join-Path $root "python\src"
$jdir = Join-Path $root "java"
$py   = Join-Path $root ".venv\Scripts\python.exe"
$res  = Join-Path $root "results"
$rows = "../data/processed/bench_rows.csv"
$url  = "http://127.0.0.1:5000/predict"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $py)) { throw "venv python not found at $py" }
New-Item -ItemType Directory -Force -Path (Join-Path $res "runs") | Out-Null
Start-Transcript -Path (Join-Path $res "runs\run_$stamp.log") | Out-Null

function Pinned([string]$cmdline) {
    # P-cores only (logical 0-3 on i5-1235U), high priority, wait for exit
    cmd /c "start `"`" /b /wait /high /affinity F $cmdline"
}

function Stop-Server {
    $c = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
    if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }
    Start-Sleep -Seconds 2
}

function Start-Server([string]$serverArgs) {
    Stop-Server
    Write-Host "`n=== starting server: $serverArgs ==="
    Start-Process -FilePath "cmd.exe" -WorkingDirectory $src `
        -ArgumentList "/c start `"SERVER`" /high /affinity F `"$py`" s04_serve.py $serverArgs"
    for ($i = 0; $i -lt 60; $i++) {
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:5000/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
            Write-Host "server is up"
            return
        } catch { Start-Sleep -Seconds 1 }
    }
    throw "server did not come up within 60 s"
}

try {
    # ---------------- machine warm-up + Python in-process ----------------
    Push-Location $src
    Write-Host "`n=== machine warm-up (discarded) ==="
    Pinned "`"$py`" s05_bench_inprocess.py --n 200"
    Write-Host "`n=== Python in-process arms ==="
    Pinned "`"$py`" s05_bench_inprocess.py"
    Pop-Location

    # ---------------- JVM in-process ----------------
    Push-Location $jdir
    Write-Host "`n=== JVM in-process arms ==="
    Pinned "java -Xmx4g -jar target\bench.jar latency --arm smile --label jvm_smile_inprocess --rows $rows --model ../results/models/rf_smile.ser --out ../results"
    Pinned "java -Xmx4g -jar target\bench.jar latency --arm onnx --label jvm_onnx_inprocess --rows $rows --model ../results/models/rf_sklearn.onnx --out ../results"
    Pop-Location

    # ---------------- B1: old-paper replica ----------------
    Start-Server "--server flask-dev --backend sklearn --n-jobs -1"
    Push-Location $src
    Pinned "`"$py`" s06_bench_rest.py --label rest_flaskdev_njobsall_nokeepalive --no-keepalive --n 200"
    Pinned "`"$py`" s06_bench_rest.py --label rest_flaskdev_njobsall_nokeepalive --no-keepalive --server-desc flaskdev_sklearn_njobs-1"
    Pop-Location

    # ---------------- B2: waitress + sklearn ----------------
    Start-Server "--server waitress --backend sklearn --n-jobs 1"
    Push-Location $src
    Pinned "`"$py`" s06_bench_rest.py --label rest_waitress_sklearn_keepalive --n 200"
    Pinned "`"$py`" s06_bench_rest.py --label rest_waitress_sklearn_keepalive --server-desc waitress_sklearn_njobs1"
    Pinned "`"$py`" s06_bench_rest.py --label rest_waitress_sklearn_nokeepalive --no-keepalive --server-desc waitress_sklearn_njobs1"
    Pop-Location
    Push-Location $jdir
    Pinned "java -Xmx4g -jar target\bench.jar latency --arm rest --label jvm_rest_waitress_sklearn --rows $rows --url $url --out ../results"
    Pop-Location

    # ---------------- B3: waitress + ONNX ----------------
    Start-Server "--server waitress --backend onnx"
    Push-Location $src
    Pinned "`"$py`" s06_bench_rest.py --label rest_waitress_onnx_keepalive --n 200"
    Pinned "`"$py`" s06_bench_rest.py --label rest_waitress_onnx_keepalive --server-desc waitress_onnx"
    Pop-Location
    Push-Location $jdir
    Pinned "java -Xmx4g -jar target\bench.jar latency --arm rest --label jvm_rest_waitress_onnx --rows $rows --url $url --out ../results"
    Pop-Location
}
finally {
    Stop-Server
    # archive this run's raw files so the next repeat does not overwrite them
    $dest = Join-Path $res "runs\$stamp"
    if (Test-Path (Join-Path $res "raw")) {
        Copy-Item -Recurse -Force (Join-Path $res "raw") $dest
        Write-Host "`nraw results archived to $dest"
    }
    Stop-Transcript | Out-Null
}