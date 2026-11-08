@echo off
REM ============================================================================
REM  Car.Ma._Cell - WINDOWS installer (setup only)
REM ----------------------------------------------------------------------------
REM  Double-click this file ONCE to set everything up. It will:
REM    1. Install Python 3.12 automatically (per-user, no admin required)
REM       if Python is not already present on this PC.
REM    2. Create a local virtual environment in .venv\
REM    3. Install every required scientific package (numpy, scipy, ...)
REM
REM  After this finishes, start the program by double-clicking:
REM        Avvia_Windows.bat
REM
REM  Nothing is installed system-wide. To "uninstall" delete this folder.
REM ============================================================================

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "APP_NAME=Car.Ma._Cell"
set "VENV=.venv"
set "VENV_PY=!VENV!\Scripts\python.exe"
set "REQS=requirements.txt"
set "STAMP=!VENV!\.carma_installed"

REM Pinned Python version for the offline fallback installer.
set "PY_VER=3.12.7"

REM Pick the right installer for this PC's CPU architecture
REM (Surface / Copilot+ PCs ship with ARM64 Windows).
set "PY_ARCH=amd64"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "PY_ARCH=arm64"
if /I "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "PY_ARCH=arm64"

set "PY_FILE=python-!PY_VER!-!PY_ARCH!.exe"
set "PY_URL=https://www.python.org/ftp/python/!PY_VER!/!PY_FILE!"
set "PY_DL=%TEMP%\!PY_FILE!"

REM ---------------------------------------------------------------------------
REM Fast path: already installed -> still re-sync dependencies so a code
REM update that adds a new package (e.g. openpyxl) is picked up on re-run.
REM ---------------------------------------------------------------------------
if exist "!VENV_PY!" if exist "!STAMP!" (
    call :banner "Already installed - checking dependencies..."
    "!VENV_PY!" -m pip install -r "!REQS!"
    if errorlevel 1 (
        echo [WARN] Could not update dependencies. If the program misbehaves,
        echo        delete the ".venv" folder and run this installer again.
    ) else (
        echo Dependencies are up to date.
    )
    echo To start the program, double-click  Avvia_Windows.bat
    echo.
    pause
    endlocal
    exit /b 0
)

call :banner "[1/3] Looking for a Python 3.9+ interpreter..."

set "PY_EXE="
call :locate_python
if defined PY_EXE goto :have_python

REM -------------------------------------------------------------------------
REM No usable Python found - install one automatically.
REM -------------------------------------------------------------------------
call :banner "[2/3] Python not found - installing Python !PY_VER! (per-user)..."

REM ---- Method A: winget (Windows 10 1809+ / Windows 11, no admin needed) ---
where winget >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    echo Trying winget...
    winget install --id Python.Python.3.12 --source winget --silent ^
        --accept-source-agreements --accept-package-agreements --scope user
    REM Re-scan PATH after winget install.
    call :locate_python
)

REM ---- Method B: official python.org installer (silent, per-user) ----------
if not defined PY_EXE (
    echo Downloading official installer from python.org...
    echo URL: !PY_URL!
    call :download "!PY_URL!" "!PY_DL!"
    if not exist "!PY_DL!" (
        echo.
        echo [ERROR] Could not download the Python installer.
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )

    echo Running silent installer ^(no admin required, per-user install^)...
    REM start /wait guarantees the installer is *fully* done before we continue.
    start "" /wait "!PY_DL!" /quiet InstallAllUsers=0 PrependPath=1 ^
        Include_test=0 Include_launcher=1 Include_pip=1 Include_tcltk=1 ^
        SimpleInstall=1 SimpleInstallDescription="Car.Ma._Cell setup"
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [ERROR] The Python installer reported an error ^(code !ERRORLEVEL!^).
        pause
        exit /b 1
    )
    del /q "!PY_DL!" >nul 2>nul

    call :locate_python
)

if not defined PY_EXE (
    echo.
    echo [ERROR] Python was installed but could not be located afterwards.
    echo Please close this window, open a NEW Command Prompt and double-click
    echo this file again.
    pause
    exit /b 1
)

:have_python
echo Using Python: "!PY_EXE!"

REM ---------------------------------------------------------------------------
REM Create venv and install dependencies
REM ---------------------------------------------------------------------------
call :banner "[3/3] Preparing local environment & installing packages..."

if not exist "!VENV_PY!" (
    "!PY_EXE!" -m venv "!VENV!"
    if not exist "!VENV_PY!" (
        echo.
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

"!VENV_PY!" -m pip install --upgrade pip setuptools wheel
"!VENV_PY!" -m pip install -r "!REQS!"
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)

> "!STAMP!" echo !APP_NAME! installed on %DATE% %TIME%

call :banner "Setup complete!"
echo Installation finished successfully.
echo.
echo To start the program now ^(and every time from now on^),
echo double-click:   Avvia_Windows.bat
echo.
pause
endlocal
exit /b 0


REM ===========================================================================
REM  Subroutines
REM ===========================================================================

:banner
echo.
echo ============================================================
echo  %APP_NAME% - %~1
echo ============================================================
exit /b 0


:locate_python
REM Try (in order): py launcher, python, python3, common install dirs.
REM Skips the Microsoft Store stub (it never reports a real sys.executable).
set "PY_EXE="

where py >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY_EXE=%%P"
)

if not defined PY_EXE (
    for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY_EXE=%%P"
)
if not defined PY_EXE (
    for /f "delims=" %%P in ('python3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY_EXE=%%P"
)

if not defined PY_EXE call :tryexe "%LocalAppData%\Programs\Python\Python313\python.exe"
if not defined PY_EXE call :tryexe "%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PY_EXE call :tryexe "%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PY_EXE call :tryexe "%LocalAppData%\Programs\Python\Python310\python.exe"
if not defined PY_EXE call :tryexe "%LocalAppData%\Programs\Python\Python39\python.exe"
if not defined PY_EXE call :tryexe "%USERPROFILE%\anaconda3\python.exe"
if not defined PY_EXE call :tryexe "%USERPROFILE%\miniconda3\python.exe"
if not defined PY_EXE call :tryexe "%ProgramData%\Anaconda3\python.exe"
if not defined PY_EXE call :tryexe "%ProgramData%\Miniconda3\python.exe"
if not defined PY_EXE call :tryexe "C:\Python313\python.exe"
if not defined PY_EXE call :tryexe "C:\Python312\python.exe"
if not defined PY_EXE call :tryexe "C:\Python311\python.exe"
if not defined PY_EXE call :tryexe "C:\Python310\python.exe"

REM Verify it is >= 3.9; otherwise discard.
if defined PY_EXE (
    "!PY_EXE!" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>nul
    if !ERRORLEVEL! NEQ 0 set "PY_EXE="
)
exit /b 0


:tryexe
if exist "%~1" (
    "%~1" -c "import sys" >nul 2>nul
    if !ERRORLEVEL! EQU 0 set "PY_EXE=%~1"
)
exit /b 0


:download
REM Usage: call :download <url> <outfile>
REM Tries curl first (built-in on Windows 10 1803+); falls back to PowerShell
REM (works on Windows 7/8.1/10/11). Removes a partial file on failure.
set "_DL_URL=%~1"
set "_DL_OUT=%~2"
if exist "!_DL_OUT!" del /q "!_DL_OUT!" >nul 2>nul

where curl >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    curl -L --fail --retry 3 --retry-delay 2 -o "!_DL_OUT!" "!_DL_URL!"
    if exist "!_DL_OUT!" exit /b 0
)

where powershell >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
        "try { Invoke-WebRequest -UseBasicParsing -Uri '!_DL_URL!' -OutFile '!_DL_OUT!' } " ^
        "catch { exit 1 }"
    if exist "!_DL_OUT!" exit /b 0
)

if exist "!_DL_OUT!" del /q "!_DL_OUT!" >nul 2>nul
exit /b 1
