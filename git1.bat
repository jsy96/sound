@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ROOT=%~dp0"
set "REMOTE_URL=https://github.com/jsy96/sound.git"
set "COMMIT_MESSAGE=%~1"
if not defined COMMIT_MESSAGE set "COMMIT_MESSAGE=Publish project"

 echo ============================================================
 echo Git publish helper
 echo Project directory: %CD%
 echo Remote repository: %REMOTE_URL%
 echo ============================================================

where git >nul 2>nul
if errorlevel 1 (
  echo ERROR: Git was not found on PATH.
  echo Install Git for Windows and run this file again.
  pause
  exit /b 1
)

if not exist "%ROOT%.git" (
  echo Initializing local Git repository...
  git init
  if errorlevel 1 goto failed
) else (
  echo Local Git repository already exists.
)

git branch -M main
if errorlevel 1 goto failed

for /f "tokens=*" %%R in ('git remote get-url origin 2^>nul') do set "CURRENT_REMOTE=%%R"
if defined CURRENT_REMOTE (
  if /I not "%CURRENT_REMOTE%"=="%REMOTE_URL%" (
    echo Updating origin remote URL...
    git remote set-url origin "%REMOTE_URL%"
    if errorlevel 1 goto failed
  )
) else (
  echo Adding origin remote...
  git remote add origin "%REMOTE_URL%"
  if errorlevel 1 goto failed
)

echo Staging files allowed by .gitignore...
git add .
if errorlevel 1 goto failed

for /f "delims=" %%F in ('git diff --cached --name-only -- data models outputs .venv __pycache__ 2^>nul') do (
  echo ERROR: A local-only path is staged: %%F
  echo Remove it from the index before publishing.
  git reset --quiet
  goto failed
)

git diff --cached --quiet
if not errorlevel 1 (
  echo No changes are staged. Nothing to commit.
  git status --short
  pause
  exit /b 0
)

echo Staged files:
git diff --cached --name-status

echo Creating commit...
git commit -m "%COMMIT_MESSAGE%"
if errorlevel 1 goto failed

echo Pushing to origin/main...
git push -u origin main
if errorlevel 1 goto failed

echo.
echo Publish completed successfully.
pause
exit /b 0

:failed
echo.
echo Publish failed. Review the error above.
pause
exit /b 1
