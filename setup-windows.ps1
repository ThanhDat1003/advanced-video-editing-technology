$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectDirectory

foreach ($commandName in @("py", "ffmpeg", "ffprobe", "nvidia-smi")) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Khong tim thay '$commandName' trong PATH. Hay cai prerequisite trong README truoc."
    }
}

$pythonVersion = & py -3.11 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Python: $pythonVersion"
& nvidia-smi

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.lock
& .\.venv\Scripts\python.exe -m pip install -e ".[test]" --no-deps
& .\.venv\Scripts\python.exe -m pytest

Write-Host ""
Write-Host "Cai dat hoan tat. Thu lenh:"
Write-Host '.\.venv\Scripts\transcript-video.exe transcribe "D:\Videos\video.mp4" -o "D:\Transcripts"'

