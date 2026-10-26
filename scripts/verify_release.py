#!/usr/bin/env python3
"""Kiểm tra mọi file được ghi trong RELEASE_SHA256.csv sau khi giải nén."""
import csv
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
manifest = root / "RELEASE_SHA256.csv"
errors = []
count = 0
with manifest.open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        rel = row["relative_path"]
        path = root / rel
        count += 1
        if not path.is_file():
            errors.append(f"MISSING: {rel}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as data:
            for block in iter(lambda: data.read(1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            errors.append(f"MISMATCH: {rel}")
print(f"Checked {count} file(s); errors: {len(errors)}")
for issue in errors:
    print(issue)
raise SystemExit(1 if errors else 0)
