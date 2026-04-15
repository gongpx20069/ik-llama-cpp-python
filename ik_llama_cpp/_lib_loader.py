"""Shared library loader for ik_llama.cpp."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path


def _lib_names() -> list[str]:
    system = platform.system()
    if system == "Windows":
        return ["llama.dll"]
    elif system == "Darwin":
        return ["libllama.dylib"]
    else:
        return ["libllama.so"]


def load_shared_library() -> ctypes.CDLL:
    """Find and load the ik_llama shared library.

    Search order:
    1. ``IK_LLAMA_CPP_LIB_PATH`` env var (exact path to .dll/.so/.dylib)
    2. ``ik_llama_cpp/lib/`` directory next to this file (pip-installed)
    """
    # 1. Explicit override
    override = os.environ.get("IK_LLAMA_CPP_LIB_PATH")
    if override:
        p = Path(override)
        if p.is_file():
            return _load(p)
        raise FileNotFoundError(f"IK_LLAMA_CPP_LIB_PATH points to missing file: {p}")

    # 2. Package lib/ directory
    lib_dir = Path(__file__).parent / "lib"
    for name in _lib_names():
        candidate = lib_dir / name
        if candidate.exists():
            return _load(candidate)

    raise FileNotFoundError(
        f"Cannot find ik_llama shared library. Searched: {lib_dir}. "
        "Set IK_LLAMA_CPP_LIB_PATH or rebuild with: pip install -e ."
    )


def _load(path: Path) -> ctypes.CDLL:
    # On Windows, add the library directory to DLL search path
    if platform.system() == "Windows":
        os.add_dll_directory(str(path.parent))
    return ctypes.CDLL(str(path))
