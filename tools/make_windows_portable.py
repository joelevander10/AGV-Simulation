"""Build AGV_Simulator_Windows.zip: the simulator + its own Python, for Windows PCs without Python.

    python3 tools/make_windows_portable.py

Needs tools/windows_portable/python-win-runtime.zip (a portable 64-bit Windows Python 3.12 with numpy,
matplotlib, pillow, pyyaml and flask already inside - no installer, no admin rights). The zip is made
from the project as it is now (code, configs, routes, maps, sequences); run it again after changes.

On the Windows PC: unzip, double-click "Start Simulator.bat".
"""
from __future__ import annotations

import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "tools", "windows_portable", "python-win-runtime.zip")
OUT = os.path.join(ROOT, "AGV_Simulator_Windows.zip")
TOP = "AGV_Simulator_Windows"

# not copied: other launchers, backups, caches, old results (made again on the PC), big zips, the runtime
SKIP_DIRS = {".backup", "__pycache__", ".pytest_cache", ".venv", "venv", "_upd", "results_demo", "results", ".git",
             os.path.join("tools", "windows_portable")}
SKIP_FILES = {"AGV_Simulator.zip", "AGV_Simulator_Windows.zip", "Start Simulator (Windows).bat",
              "Start Simulator.command", "start_simulator.sh", "START HERE.txt", ".DS_Store"}

BAT = r"""@echo off
title AGV Simulator
cd /d "%~dp0"
if not exist "%~dp0python\python.exe" (
  echo The python folder is missing. Unzip the WHOLE zip first ^(right-click - Extract All^), then start again.
  pause
  exit /b 1
)
echo AGV Simulator is starting - your browser opens by itself.
echo Keep this window open while you work. Close it to stop the simulator.
echo.
"%~dp0python\python.exe" "%~dp0app.py"
if errorlevel 1 pause
"""

README = """AGV SIMULATOR - WINDOWS (NO PYTHON NEEDED)
==========================================

Python and everything the simulator needs are inside this folder (python\\).
Nothing is installed on the PC and no admin rights are needed.

1. Unzip: right-click the zip -> Extract All... (do not start it from inside the zip).
   Put the folder somewhere normal, e.g. Documents or Desktop.

2. Double-click  "Start Simulator.bat"
   Your browser opens the simulator. Keep the black window open while you work;
   close it to stop the simulator.

   If Windows says "Windows protected your PC": click "More info" -> "Run anyway".
   (Or before unzipping: right-click the zip -> Properties -> tick "Unblock" -> OK.)
   If the company antivirus blocks python\\python.exe, ask IT to allow this folder.

Your routes, vehicles, floor plans and sequences are in routes\\, configs\\, maps\\ and
sequences\\; results go to results\\. Full manual: README.md. Click "? Help" in the app.

Made {date}. Updating: copy a newer AGV_Simulator_Windows.zip, or copy the new agv_sim\\,
web\\, app.py and main.py over these ones (keep the python\\ folder and your own files).
"""


def _skip_dir(rel):
    rel = rel.replace("\\", "/")
    return any(rel == d.replace("\\", "/") or rel.split("/")[-1] == d for d in SKIP_DIRS)


def main():
    if not os.path.exists(RUNTIME):
        sys.exit(f"Missing {os.path.relpath(RUNTIME, ROOT)} - the portable Windows Python (ask for it again).")
    tmp = OUT + ".part"
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(ROOT):
            rel_root = os.path.relpath(root, ROOT)
            dirs[:] = sorted(d for d in dirs if not _skip_dir(os.path.normpath(os.path.join(rel_root, d))))
            for f in sorted(files):
                if f in SKIP_FILES or f.endswith((".part", ".pyc")):
                    continue
                p = os.path.join(root, f)
                rel = os.path.normpath(os.path.join(rel_root, f)).replace("\\", "/")
                z.write(p, f"{TOP}/{rel}")
                n += 1
        z.writestr(f"{TOP}/Start Simulator.bat", BAT.replace("\n", "\r\n"))
        z.writestr(f"{TOP}/START HERE.txt", README.format(date=time.strftime("%Y-%m-%d")).replace("\n", "\r\n"))
        with zipfile.ZipFile(RUNTIME) as rz:
            for info in rz.infolist():
                if info.is_dir():
                    continue
                z.writestr(zipfile.ZipInfo(f"{TOP}/{info.filename}", date_time=info.date_time), rz.read(info),
                           compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    os.replace(tmp, OUT)
    print(f"Saved {os.path.relpath(OUT, ROOT)}: {n} project files + Windows Python, "
          f"{os.path.getsize(OUT) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
