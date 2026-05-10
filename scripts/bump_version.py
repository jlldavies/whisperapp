#!/usr/bin/env python3
"""
Bump the version string in all files that hardcode it.

Usage:
    python scripts/bump_version.py 1.2.0
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

REPLACEMENTS = [
    # (file, pattern, replacement-template)
    (
        "pyproject.toml",
        r'^version = "[\d.]+"',
        'version = "{version}"',
    ),
    (
        "whisperapp/__init__.py",
        r'^__version__ = "[\d.]+"',
        '__version__ = "{version}"',
    ),
    (
        "whisperapp.spec",
        r'"CFBundleVersion": "[\d.]+"',
        '"CFBundleVersion": "{version}"',
    ),
    (
        "whisperapp.spec",
        r'"CFBundleShortVersionString": "[\d.]+"',
        '"CFBundleShortVersionString": "{version}"',
    ),
    (
        "whisperapp.spec",
        r"'CFBundleVersion': '[\d.]+'",
        "'CFBundleVersion': '{version}'",
    ),
    (
        "whisperapp.spec",
        r"'CFBundleShortVersionString': '[\d.]+'",
        "'CFBundleShortVersionString': '{version}'",
    ),
    (
        "installer/windows.iss",
        r'^AppVersion=[\d.]+',
        "AppVersion={version}",
    ),
    (
        "installer/windows.iss",
        r'^#define AppVersion "[\d.]+"',
        '#define AppVersion "{version}"',
    ),
    (
        "whisperapp/server.py",
        r'FastAPI\(title="WhisperApp API", version="[\d.]+"\)',
        'FastAPI(title="WhisperApp API", version="{version}")',
    ),
    (
        "whisperapp/static/js/shell.js",
        r'<div class="wa-brand-sub">v[\d.]+</div>',
        '<div class="wa-brand-sub">v{version}</div>',
    ),
]


def bump(version: str) -> None:
    # Basic semver validation
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        sys.exit(f"Error: version must be X.Y.Z (got {version!r})")

    changed = []
    for rel_path, pattern, template in REPLACEMENTS:
        path = ROOT / rel_path
        if not path.exists():
            print(f"  skip  {rel_path} (not found)")
            continue
        text = path.read_text(encoding="utf-8")
        replacement = template.format(version=version)
        new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if n:
            path.write_text(new_text, encoding="utf-8")
            changed.append((rel_path, n))
        else:
            print(f"  warn  {rel_path}: pattern not matched — {pattern!r}")

    if changed:
        print(f"\nBumped to {version}:")
        for f, n in changed:
            print(f"  {f} ({n} replacement{'s' if n > 1 else ''})")
    else:
        print("No files changed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <version>   e.g. 1.2.0")
    bump(sys.argv[1])
