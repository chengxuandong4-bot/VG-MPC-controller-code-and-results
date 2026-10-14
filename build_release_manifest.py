#!/usr/bin/env python3
"""Build or verify the file inventory for this source release."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.csv"
SUMS = ROOT / "RELEASE_SHA256SUMS.txt"
EXCLUDED = {MANIFEST.name, SUMS.name}
FORBIDDEN_PARTS = {"__pycache__", ".venv"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory() -> list[tuple[str, int, str]]:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or rel.as_posix() in EXCLUDED:
            continue
        if FORBIDDEN_PARTS.intersection(rel.parts) or path.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Unwanted generated file in release tree: {rel}")
        rows.append((rel.as_posix(), path.stat().st_size, sha256_file(path)))
    return rows


def render(rows: list[tuple[str, int, str]]) -> tuple[str, str]:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("relative_path", "bytes", "sha256"))
    writer.writerows(rows)
    sums = "".join(f"{digest}  {rel}\n" for rel, _, digest in rows)
    return buffer.getvalue(), sums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()
    manifest, sums = render(inventory())
    if args.check:
        if not MANIFEST.is_file() or not SUMS.is_file():
            raise SystemExit("Release manifest or checksum file is missing")
        if MANIFEST.read_text(encoding="utf-8") != manifest:
            raise SystemExit("MANIFEST.csv is out of date")
        if SUMS.read_text(encoding="utf-8") != sums:
            raise SystemExit("RELEASE_SHA256SUMS.txt is out of date")
        print("release_manifest: PASS")
    else:
        MANIFEST.write_text(manifest, encoding="utf-8")
        SUMS.write_text(sums, encoding="utf-8")
        print("release_manifest: written")


if __name__ == "__main__":
    main()
