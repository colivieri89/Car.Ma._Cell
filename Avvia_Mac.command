#!/usr/bin/env bash
# ============================================================================
#  Car.Ma._Cell - macOS launcher (starts the program)
# ----------------------------------------------------------------------------
#  Double-click this file in the Finder to open the Car.Ma._Cell graphical
#  interface.
#
#  The first time, you must run  Install_Mac.command  once to set everything
#  up. After that, just double-click this file whenever you want to use the
#  program.
#
#  GATEKEEPER: the first time, right-click the file -> Open -> Open in the
#              dialog. macOS will remember the choice afterwards.
# ============================================================================

set -u

# When launched from the Finder the PATH can be minimal.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

APP_NAME="Car.Ma._Cell"
VENV=".venv"
VENV_PY="$VENV/bin/python"
STAMP="$VENV/.carma_installed"

pause_on_exit() {
    echo ""
    echo "Press Return to close this window..."
    # shellcheck disable=SC2034
    read -r _dummy
}

# ----------------------------------------------------------------------------
# Make sure the program has been installed first.
# ----------------------------------------------------------------------------
if [ ! -x "$VENV_PY" ] || [ ! -f "$STAMP" ]; then
    echo ""
    echo "============================================================"
    echo " $APP_NAME is not installed yet"
    echo "============================================================"
    echo ""
    echo "Please run  Install_Mac.command  first to set things up,"
    echo "then double-click this file again."
    echo ""
    pause_on_exit
    exit 1
fi

echo ""
echo "============================================================"
echo " $APP_NAME - starting..."
echo "============================================================"
"$VENV_PY" gui.py
rc=$?
if [ "$rc" -ne 0 ]; then
    echo ""
    echo "[$APP_NAME] The application exited with code $rc."
    pause_on_exit
fi
exit "$rc"
