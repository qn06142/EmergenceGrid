"""Rebuild cpp_sim.pyd with MSVC (VS 2022 BuildTools). Run from repo root.

This script is portable: it derives all paths from the repository root and the
active Python environment, so it works on any machine with VS2022 BuildTools
and pybind11 installed in the venv (``pip install pybind11``).

Usage:
    python build_cpp.py
"""
import subprocess, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# MSVC toolchain (adjust if your VS install lives elsewhere).
VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Derive Python include/lib from the *running* interpreter (works for uv/venv).
PYINC = os.path.join(sys.base_prefix, "Include") if hasattr(sys, "base_prefix") else sys.prefix
PYLIB = os.path.join(sys.base_prefix, "libs") if hasattr(sys, "base_prefix") else os.path.join(sys.prefix, "libs")

# pybind11 include dir from the active environment's site-packages.
def _find_pybind():
    for base in (sys.base_prefix, sys.prefix):
        cand = os.path.join(base, "Lib", "site-packages", "pybind11", "include")
        if os.path.isdir(cand):
            return cand
    # fall back to the venv created next to this repo
    cand = os.path.join(ROOT, ".venv", "Lib", "site-packages", "pybind11", "include")
    if os.path.isdir(cand):
        return cand
    raise FileNotFoundError("pybind11 not found in site-packages; run `pip install pybind11`")

PYBIND = _find_pybind()
SRC = os.path.join(ROOT, "cpp", "sim.cpp")
OUT = os.path.join(ROOT, "cpp_sim.pyd")

cmd = (
    f'call "{VCVARS}" && '
    f'cl /O2 /LD /std:c++17 /EHsc '
    f'/I"{PYINC}" /I"{PYBIND}" '
    f'"{SRC}" '
    f'/link /LIBPATH:"{PYLIB}" python3.lib '
    f'/OUT:"{OUT}"'
)
print("BUILDING:", cmd)
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout)
print(r.stderr)
print("EXITCODE=", r.returncode)
sys.exit(r.returncode)
