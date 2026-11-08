#!/usr/bin/env bash
# ============================================================================
#  Car.Ma._Cell - macOS installer (setup only)
# ----------------------------------------------------------------------------
#  Double-click this file in the Finder ONCE to set everything up. It will:
#    1. Install Python 3.12 automatically (asks for your macOS password once)
#       if Python is not already present.
#    2. Create a local virtual environment in .venv/
#    3. Install every required scientific package (numpy, scipy, ...)
#
#  After this finishes, start the program by double-clicking:
#        Avvia_Mac.command
#
#  Nothing is installed system-wide by Car.Ma._Cell itself; the only system
#  install is Python from python.org (the same file you would download by
#  hand), which is needed for the Tkinter GUI to work.
#
#  Works on macOS 11 (Big Sur) or newer, both Intel and Apple Silicon.
#  GATEKEEPER: the first time, right-click the file -> Open -> Open in the
#              dialog. macOS will remember the choice afterwards.
# ============================================================================

set -u

# When launched from the Finder the PATH can be minimal. Guarantee the
# essential system directories are visible so curl, osascript, installer,
# etc. are always found.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

APP_NAME="Car.Ma._Cell"
VENV=".venv"
VENV_PY="$VENV/bin/python"
REQS="requirements.txt"
STAMP="$VENV/.carma_installed"

# Pinned Python version used when we need to auto-install.
PY_VER="3.12.7"
PY_PKG_URL="https://www.python.org/ftp/python/${PY_VER}/python-${PY_VER}-macos11.pkg"
PY_PKG_DL="/tmp/python-${PY_VER}-macos11.pkg"

banner() {
    echo ""
    echo "============================================================"
    echo " $APP_NAME - $1"
    echo "============================================================"
}

pause_on_exit() {
    echo ""
    echo "Press Return to close this window..."
    # shellcheck disable=SC2034
    read -r _dummy
}

# Make sure that, if the user double-clicked the file and something fails,
# they still see the error before the Terminal window closes.
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo ""; echo "[$APP_NAME] Aborted (exit $rc)."; pause_on_exit; fi' EXIT

# ----------------------------------------------------------------------------
# Fast path: already installed -> still re-sync dependencies so that a code
# update which adds a new package (e.g. openpyxl) is picked up on re-run.
# ----------------------------------------------------------------------------
if [ -x "$VENV_PY" ] && [ -f "$STAMP" ]; then
    banner "Already installed - checking dependencies..."
    if "$VENV_PY" -m pip install -r "$REQS"; then
        echo "Dependencies are up to date."
    else
        echo "[WARN] Could not update dependencies. If the program misbehaves,"
        echo "       delete the '.venv' folder and run this installer again."
    fi
    echo "To start the program, double-click  Avvia_Mac.command"
    trap - EXIT
    pause_on_exit
    exit 0
fi

# ----------------------------------------------------------------------------
# Locate (or auto-install) a usable Python 3.9+ with Tkinter
# ----------------------------------------------------------------------------
PY_EXE=""

check_python() {
    local cand="$1"
    [ -x "$cand" ] || return 1
    "$cand" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >/dev/null 2>&1 || return 1
    "$cand" -c "import tkinter" >/dev/null 2>&1 || return 1
    PY_EXE="$cand"
    return 0
}

try_cmd() {
    [ -n "$PY_EXE" ] && return 0
    local cand="$1"
    if command -v "$cand" >/dev/null 2>&1; then
        check_python "$(command -v "$cand")" || true
    fi
}

try_path() {
    [ -n "$PY_EXE" ] && return 0
    check_python "$1" || true
}

locate_python() {
    PY_EXE=""
    # python.org official installer (always ships with Tkinter)
    try_path "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
    try_path "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
    try_path "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    try_path "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
    try_path "/Library/Frameworks/Python.framework/Versions/3.9/bin/python3"

    # Homebrew - Apple Silicon then Intel
    try_path "/opt/homebrew/bin/python3.13"
    try_path "/opt/homebrew/bin/python3.12"
    try_path "/opt/homebrew/bin/python3.11"
    try_path "/opt/homebrew/bin/python3.10"
    try_path "/usr/local/bin/python3.13"
    try_path "/usr/local/bin/python3.12"
    try_path "/usr/local/bin/python3.11"
    try_path "/usr/local/bin/python3.10"

    # Anything on PATH
    try_cmd python3.13
    try_cmd python3.12
    try_cmd python3.11
    try_cmd python3.10
    try_cmd python3.9
    try_cmd python3

    # Anaconda / Miniforge
    try_path "$HOME/anaconda3/bin/python3"
    try_path "$HOME/miniconda3/bin/python3"
    try_path "$HOME/miniforge3/bin/python3"
    try_path "$HOME/mambaforge/bin/python3"
}

banner "[1/3] Looking for a Python 3.9+ interpreter with Tkinter..."
locate_python

if [ -z "$PY_EXE" ]; then
    banner "[2/3] Python not found - installing Python $PY_VER from python.org..."
    echo "Downloading: $PY_PKG_URL"

    if [ ! -x /usr/bin/curl ] && ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] 'curl' is missing. Please install Python manually from"
        echo "        https://www.python.org/downloads/macos/"
        exit 1
    fi

    CURL_BIN="$(command -v curl || echo /usr/bin/curl)"
    if ! "$CURL_BIN" -L --fail --retry 3 --retry-delay 2 -o "$PY_PKG_DL" "$PY_PKG_URL"; then
        echo "[ERROR] Could not download the Python installer."
        echo "        Check your internet connection and try again."
        exit 1
    fi

    echo ""
    echo "macOS will now ask for your password to install Python system-wide."
    echo "(This is the same install you would do by hand from python.org.)"
    echo ""

    # Run 'installer' with admin privileges via a native macOS password prompt.
    if ! osascript -e "do shell script \"/usr/sbin/installer -pkg '$PY_PKG_DL' -target /\" with administrator privileges with prompt \"$APP_NAME needs to install Python $PY_VER.\""; then
        echo "[ERROR] Python installation was cancelled or failed."
        exit 1
    fi

    rm -f "$PY_PKG_DL"

    locate_python
    if [ -z "$PY_EXE" ]; then
        echo "[ERROR] Python was installed but could not be located afterwards."
        echo "        Please close this window and double-click the installer again."
        exit 1
    fi
fi

echo "Using Python: $PY_EXE"

# ----------------------------------------------------------------------------
# Create venv and install dependencies
# ----------------------------------------------------------------------------
banner "[3/3] Preparing local environment & installing packages..."

if [ ! -x "$VENV_PY" ]; then
    if ! "$PY_EXE" -m venv "$VENV"; then
        echo "[ERROR] Failed to create the virtual environment."
        exit 1
    fi
fi

"$VENV_PY" -m pip install --upgrade pip setuptools wheel
if ! "$VENV_PY" -m pip install -r "$REQS"; then
    echo "[ERROR] Dependency installation failed."
    echo "        Check your internet connection and try again."
    exit 1
fi

echo "$APP_NAME installed on $(date)" > "$STAMP"

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
banner "Setup complete!"
echo "Installation finished successfully."
echo ""
echo "To start the program now (and every time from now on),"
echo "double-click:   Avvia_Mac.command"
trap - EXIT
pause_on_exit
exit 0
