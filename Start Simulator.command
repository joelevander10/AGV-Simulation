#!/bin/bash
# Double-click me (macOS). First start installs everything into a local .venv folder.
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Get it from https://www.python.org/downloads/ and try again."
  read -r -p "Press Enter to close..."; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "First start: setting up (takes a minute)..."
  python3 -m venv .venv || { read -r -p "Could not create .venv. Press Enter..."; exit 1; }
  .venv/bin/python -m pip install --upgrade pip -q
fi
.venv/bin/python -m pip install -q -r requirements.txt || { read -r -p "Install failed (internet?). Press Enter..."; exit 1; }
.venv/bin/python app.py
