"""Allow only one recipe run per user on this machine."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path


@contextmanager
def run_lock():
    # Independent of checkout, recipe paths, and per-run TMPDIR settings.
    registry = Path('/tmp') / f'dw93net-run-locks-{os.getuid()}'
    registry.mkdir(mode=0o700, exist_ok=True)
    descriptor = os.open(registry / 'run.lock', os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another recipe run is already active for this user on this machine') from error
        yield (descriptor,)
    finally:
        # Close rather than LOCK_UN: inherited descriptors keep background
        # generation protected if its parent dies. Never unlink the lock file;
        # a new inode could allow two processes to acquire different locks.
        os.close(descriptor)
