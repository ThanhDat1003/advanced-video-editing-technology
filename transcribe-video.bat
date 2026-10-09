@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if "%~1"=="" (
  echo Hay keo-tha mot hoac nhieu video vao file nay.
  pause
  popd
  exit /b 2
)
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m transcript_video.cli transcribe %*
) else (
  py -3.11 -m transcript_video.cli transcribe %*
)
if errorlevel 1 pause
popd
endlocal
