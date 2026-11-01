@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "LOG_FILE=%ROOT%start.log"
set "VENV_PYTHON=%ROOT%.venv\Scripts\python.exe"
set "PATH=%ROOT%.venv\Scripts;%PATH%"
set "HF_HUB_DOWNLOAD_TIMEOUT=600"
set "HF_HUB_ETAG_TIMEOUT=120"
set "HF_HUB_DOWNLOAD_RETRIES=10"
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://huggingface.co"
if not defined VOICE_MODEL_ENDPOINTS set "VOICE_MODEL_ENDPOINTS=https://huggingface.co,https://hf-mirror.com"
set "MODEL_CACHE=%ROOT%models\huggingface\hub\models--2Noise--ChatTTS"

>"%LOG_FILE%" echo [%DATE% %TIME%] Starting Voice Design server from %CD%
echo ============================================================
echo Chinese Voice Design Workbench
echo Project directory: %CD%
echo Diagnostic log: %LOG_FILE%
echo ============================================================

if not exist "%ROOT%voice_design_server.py" (
  echo ERROR: voice_design_server.py was not found in the project directory.
  >>"%LOG_FILE%" echo ERROR: voice_design_server.py was not found.
  goto failed
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ERROR: ffmpeg was not found on PATH.
  echo Install FFmpeg and add its bin directory to PATH.
  >>"%LOG_FILE%" echo ERROR: ffmpeg was not found on PATH.
  goto failed
)
ffmpeg -version >nul 2>nul
if errorlevel 1 (
  echo ERROR: ffmpeg was found but could not be executed.
  >>"%LOG_FILE%" echo ERROR: ffmpeg execution failed.
  goto failed
)

if not exist "%VENV_PYTHON%" (
  echo ERROR: project Python environment was not found:
  echo %VENV_PYTHON%
  echo Create the environment and install requirements.txt before starting.
  >>"%LOG_FILE%" echo ERROR: project Python environment was not found.
  goto failed
)

"%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
  echo ERROR: the project environment must use Python 3.11.
  >>"%LOG_FILE%" echo ERROR: project environment is not Python 3.11.
  goto failed
)

"%VENV_PYTHON%" -c "import ChatTTS, transformers; raise SystemExit(0 if int(transformers.__version__.split('.')[0]) < 5 else 1)" >nul 2>nul
if errorlevel 1 (
  echo ERROR: ChatTTS dependencies are not installed in the project environment.
  echo Install requirements.txt, then run this file again.
  >>"%LOG_FILE%" echo ERROR: ChatTTS dependencies are not installed.
  goto failed
)

if exist "%MODEL_CACHE%\refs\main" (
  "%VENV_PYTHON%" -c "from pathlib import Path; p=Path(r'%MODEL_CACHE%'); rev=(p/'refs'/'main').read_text(encoding='utf-8').strip(); raise SystemExit(0 if (p/'snapshots'/rev/'asset'/'Embed.safetensors').is_file() else 1)"
  if errorlevel 1 (
    echo WARNING: the cached ChatTTS model is incomplete.
    echo Moving the incomplete cache aside so the download can resume cleanly.
    move "%MODEL_CACHE%" "%MODEL_CACHE%.incomplete-%RANDOM%" >nul
    if errorlevel 1 (
      echo ERROR: could not move the incomplete model cache.
      >>"%LOG_FILE%" echo ERROR: could not move the incomplete model cache.
      goto failed
    )
    >>"%LOG_FILE%" echo WARNING: moved incomplete ChatTTS cache aside for a clean retry.
  )
)

>>"%LOG_FILE%" echo FFmpeg and ChatTTS dependencies are available.
echo Server URL: http://127.0.0.1:8765
echo The first generation downloads the ChatTTS model.
echo Keep this window open while using the workbench.
start "" /b powershell.exe -NoProfile -Command "Start-Sleep -Milliseconds 1200; Start-Process 'http://127.0.0.1:8765'"
powershell.exe -NoProfile -Command "& '%VENV_PYTHON%' -u '%ROOT%voice_design_server.py' --host 127.0.0.1 --port 8765 2>&1 | Tee-Object -FilePath '%LOG_FILE%' -Append; exit $LASTEXITCODE"
set "SERVER_EXIT=%ERRORLEVEL%"
>>"%LOG_FILE%" echo Server exited with code %SERVER_EXIT%.
echo.
echo Server stopped with exit code %SERVER_EXIT%.
echo Diagnostic log:
echo %LOG_FILE%
type "%LOG_FILE%"
pause
exit /b %SERVER_EXIT%

:failed
echo.
echo Startup failed. Diagnostic log:
echo %LOG_FILE%
type "%LOG_FILE%"
pause
exit /b 1
