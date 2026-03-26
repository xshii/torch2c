#!/usr/bin/env python3
"""Check project conventions — guardrails for AI-assisted development.

Checks:
  - Functions > 50 lines            (warning)
  - Files > 300 lines in torch2c/   (warning, excludes tests)
  - print() usage in torch2c/ src   (error, excludes tests/scripts)
  - .addr usage in C mock files     (error, should use .ptr)
  - Hardcoded alignment in calc_padded_size calls  (error)

Exit codes:
  0 — clean or warnings only
  1 — errors found
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH2C_DIR = ROOT / "torch2c"
NPU_MOCK_DIR = ROOT / "npu_cpu_mock"

MAX_FUNC_LINES = 50
MAX_FILE_LINES = 300

warnings: list[str] = []
errors: list[str] = []


# ── Helpers ──────────────────────────────────────────────────────


# 这些文件是用户交互/调试工具，允许 print
_PRINT_ALLOW_FILES = {"torch_debug.py", "sync_api_header.py"}


def _is_excluded_from_print_check(path: Path) -> bool:
    """排除测试、demo、脚本、调试工具文件（这些允许 print）。"""
    parts = path.parts
    return (
        "tests" in parts
        or "demo" in parts
        or "scripts" in parts
        or path.name.startswith("test_")
        or path.name == "conftest.py"
        or path.name in _PRINT_ALLOW_FILES
    )


def python_source_files() -> list[Path]:
    """torch2c/ 下的核心源码（排除 tests/demo/scripts）。"""
    return [
        p
        for p in TORCH2C_DIR.rglob("*.py")
        if not _is_excluded_from_print_check(p)
    ]


def c_mock_files() -> list[Path]:
    """Yield .c files under npu_cpu_mock/."""
    if not NPU_MOCK_DIR.exists():
        return []
    return list(NPU_MOCK_DIR.rglob("*.c"))


# ── Check: function length ───────────────────────────────────────


def check_function_length(path: Path) -> None:
    """Warn if any function/method exceeds MAX_FUNC_LINES."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # end_lineno is available in Python 3.8+
            if node.end_lineno is None:
                continue
            length = node.end_lineno - node.lineno + 1
            if length > MAX_FUNC_LINES:
                rel = path.relative_to(ROOT)
                warnings.append(
                    f"  {rel}:{node.lineno} — function '{node.name}' "
                    f"is {length} lines (limit {MAX_FUNC_LINES})"
                )


# ── Check: file length ──────────────────────────────────────────


def check_file_length(path: Path) -> None:
    """Warn if a source file exceeds MAX_FILE_LINES."""
    try:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return

    if line_count > MAX_FILE_LINES:
        rel = path.relative_to(ROOT)
        warnings.append(
            f"  {rel} — {line_count} lines (limit {MAX_FILE_LINES})"
        )


# ── Check: print() in source ────────────────────────────────────

PRINT_RE = re.compile(r"\bprint\s*\(")


def check_no_print(path: Path) -> None:
    """Error if print() is used in torch2c/ source (not tests/scripts)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return

    for lineno, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        # Skip comments and noqa-suppressed lines
        if stripped.startswith("#") or "noqa: print" in line:
            continue
        if PRINT_RE.search(line):
            rel = path.relative_to(ROOT)
            errors.append(
                f"  {rel}:{lineno} — use logger instead of print()"
            )


# ── Check: .addr in C files ─────────────────────────────────────

ADDR_RE = re.compile(r"\.addr\b")


def check_no_addr(path: Path) -> None:
    """Error if .addr is used in C mock files (should use .ptr)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return

    for lineno, line in enumerate(lines, start=1):
        if ADDR_RE.search(line):
            rel = path.relative_to(ROOT)
            errors.append(
                f"  {rel}:{lineno} — use .ptr instead of .addr"
            )


# ── Check: hardcoded alignment ──────────────────────────────────

PADDED_SIZE_RE = re.compile(r"calc_padded_size\([^)]*\(16\s*,\s*16\)")


def check_no_hardcoded_alignment(path: Path) -> None:
    """Error if calc_padded_size is called with hardcoded (16, 16)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return

    for lineno, line in enumerate(lines, start=1):
        if PADDED_SIZE_RE.search(line):
            rel = path.relative_to(ROOT)
            errors.append(
                f"  {rel}:{lineno} — do not hardcode (16, 16) in "
                f"calc_padded_size; use alignment from config"
            )


# ── Main ─────────────────────────────────────────────────────────


def main() -> int:
    py_sources = python_source_files()
    c_sources = c_mock_files()
    all_py = list(TORCH2C_DIR.rglob("*.py"))

    # Function length — check ALL .py files (including tests)
    for p in all_py:
        check_function_length(p)

    # File length — source files only (not tests)
    for p in py_sources:
        check_file_length(p)

    # print() — source files only
    for p in py_sources:
        check_no_print(p)

    # .addr — C mock files
    for p in c_sources:
        check_no_addr(p)

    # Hardcoded alignment — all Python files
    for p in all_py:
        check_no_hardcoded_alignment(p)

    # ── Report ───────────────────────────────────────────────────
    if warnings:
        print(f"\n{'='*60}")
        print(f"WARNINGS ({len(warnings)}):")
        print(f"{'='*60}")
        for w in warnings:
            print(w)

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        print(f"{'='*60}")
        for e in errors:
            print(e)
        print()
        return 1

    if not warnings:
        print("All convention checks passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
