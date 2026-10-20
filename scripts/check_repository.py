"""Dependency-free submission checks: hashes, links, syntax, and excluded content."""

import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "external", "outputs", "weights", "__pycache__"}


def check():
    errors = []
    for p in ROOT.rglob("*"):
        relative = p.relative_to(ROOT)
        if any(x in SKIP or x.startswith(".venv") for x in relative.parts) or not p.is_file():
            continue
        if p.suffix in (".pdf", ".pptx", ".ndpi", ".ndpa", ".pt", ".pth") or p.name == "AGENTS.md":
            errors.append(f"Excluded submission file: {relative}")
        if p.stat().st_size > 5_000_000:
            errors.append(f"Unexpected large file: {relative}")
        if p.suffix == ".py":
            try:
                ast.parse(p.read_text())
            except SyntaxError as e:
                errors.append(f"{relative}: {e}")
            if re.search(r"[\"\']/(?:home|mnt|rhf|projects)/", p.read_text()):
                errors.append(f"Hardcoded machine path: {relative}")
        if p.suffix == ".md":
            text = re.sub(r"```.*?```", "", p.read_text(), flags=re.S)
            for target in re.findall(r"\]\(([^\n)]+)\)", text):
                if urlsplit(target).scheme or target.startswith("#"):
                    continue
                if not (p.parent / unquote(target.split("#")[0])).exists():
                    errors.append(f"Broken link in {relative}: {target}")
    manifest = ROOT / "provenance/files.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text()):
            p = ROOT / item["file"]
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f'Package fingerprint mismatch: {item["file"]}')
    return errors


if __name__ == "__main__":
    errors = check()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Submission syntax, links, file policy, and available package fingerprints passed.")
