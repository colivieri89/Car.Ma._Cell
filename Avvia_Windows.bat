@echo off
REM ============================================================================
REM  Car.Ma._Cell - WINDOWS launcher (starts the program)
REM ----------------------------------------------------------------------------
REM  Double-click this file to open the Car.Ma._Cell graphical interface.
REM
REM  The first time, you must run  Install_Windows.bat  once to set everything
REM  up. After that, just double-click this file whenever you want to use the
REM  program.
REM ============================================================================

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "APP_NAME=Car.Ma._Cell"
set "VENV=.venv"
set "VENV_PY=!VENV!\Scripts\python.exe"
set "STAMP=!VENV!\.carma_installed"

REM ---------------------------------------------------------------------------
REM Make sure the program has been installed first.
REM ---------------------------------------------------------------------------
if not exist "!VENV_PY!" goto :not_installed
if not exist "!STAMP!"  goto :not_installed

echo.
echo ============================================================
echo  %APP_NAME% - starting...
echo ============================================================
"!VENV_PY!" gui.py
set "RC=!ERRORLEVEL!"
if !RC! NEQ 0 (
    echo.
    echo [%APP_NAME%] The application exited with code !RC!.
    pause
)
endlocal
exit /b 0

:not_installed
echo.
echo ============================================================
echo  %APP_NAME% is not installed yet
echo ============================================================
echo.
echo Please double-click  Install_Windows.bat  first to set things up,
echo then run this file again.
echo.
pause
endlocal
exit /b 1
