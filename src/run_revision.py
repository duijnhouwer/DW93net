"""One unchanged, clean Git commit for the lifetime of a mission plan run."""
from pathlib import Path
from datetime import datetime, timezone
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def revision(root=ROOT):
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def commit_timestamp(root, commit):
    """Return the commit's committer timestamp in UTC, independent of local TZ."""
    try:
        seconds = subprocess.check_output(
            ['git', '-C', str(root), 'show', '-s', '--format=%ct', commit],
            text=True, stderr=subprocess.PIPE).strip()
        return datetime.fromtimestamp(int(seconds), timezone.utc).strftime('%Y%m%d%H%M%S')
    except (subprocess.CalledProcessError, ValueError, OverflowError) as error:
        raise ValueError(f'Cannot read timestamp for Git commit {commit}') from error


def run_directory_name(mission_plan_name, commit=None, *, root=ROOT):
    """Stable output identity for one mission plan and exact source revision."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', mission_plan_name):
        raise ValueError('Mission plan filename must be a filesystem-safe network name')
    commit = revision(root) if commit is None else commit
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Git commit must be the full 40-character lowercase ID')
    return f'{mission_plan_name}-{commit_timestamp(root, commit)}-{commit}'


def committed_revision(root=ROOT, expected=None):
    status = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if status.strip():
        raise SystemExit('ERROR: Repository changed or is not clean. Commit changes before starting a new run:\n' + status)
    current = revision(root)
    if expected is not None and current != expected:
        raise SystemExit(f'ERROR: Repository commit changed: expected {expected}, found {current}')
    return current
