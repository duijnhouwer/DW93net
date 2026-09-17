"""One unchanged, clean Git commit for the lifetime of a recipe run."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def revision(root=ROOT):
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def committed_revision(root=ROOT, expected=None):
    status = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if status.strip():
        raise SystemExit('ERROR: Repository changed or is not clean. Commit changes before starting a new run:\n' + status)
    current = revision(root)
    if expected is not None and current != expected:
        raise SystemExit(f'ERROR: Repository commit changed: expected {expected}, found {current}')
    return current
