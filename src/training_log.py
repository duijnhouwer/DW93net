"""Append-only JSON Lines events shared by the runner and its generator."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import traceback


class EventLog:
    def __init__(self, path, run_id=None):
        self.path = Path(path)
        self.run_id = run_id
        if self.path.name == 'training_log.jsonl' and self.path.with_suffix('.json').exists():
            raise ValueError('Obsolete training_log.json found; use a fresh output folder or '
                             'archive the old run before starting. JSON log migration is not supported.')

    def read(self):
        if not self.path.exists():
            return []
        events = []
        with self.path.open('rb') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            for number, line in enumerate(stream, 1):
                # A newline commits a record. A killed writer may leave a tail;
                # the next append removes only that unfinished record.
                if not line.endswith(b'\n'):
                    break
                try:
                    value = json.loads(line)
                    if (not isinstance(value, dict) or value.get('version') != 1
                            or not isinstance(value.get('event'), str) or 'run_id' not in value):
                        raise ValueError('Invalid event schema')
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f'Invalid training event at {self.path}:{number}') from error
                events.append(value)
        return events

    def events(self, event=None):
        return [row for row in self.read()
                if row.get('run_id') == self.run_id
                and (event is None or row.get('event') == event)]

    def append(self, event, **details):
        def encode(kind, fields):
            return (json.dumps(dict(
                version=1, timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                event=kind, run_id=self.run_id, pid=os.getpid(), **fields),
                ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')

        encoded = encode(event, details)  # Reject invalid values before modifying the log.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with self.path.open('a+b') as stream:
            # The log inode is stable: never replace/unlink it during a run.
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0, os.SEEK_END)
            end = position = stream.tell()
            boundary = 0
            while position:
                start = max(0, position - 4096)
                stream.seek(start)
                chunk = stream.read(position - start)
                newline = chunk.rfind(b'\n')
                if newline >= 0:
                    boundary = start + newline + 1
                    break
                position = start
            if boundary != end:
                stream.truncate(boundary)
                stream.write(encode('log_tail_recovered', {'discarded_bytes': end - boundary}))
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if new_file:
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class _Transcript:
    def __init__(self, console, log, label):
        self.console, self.log, self.label = console, log, label
        self.lock = threading.RLock()
        self.pending = ''

    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.pending += text
            while '\n' in self.pending:
                line, self.pending = self.pending.split('\n', 1)
                if line:
                    self.log.append('console', level=self.label, message=line)
        return len(text)

    def flush(self):
        self.console.flush()

    def finish(self):
        with self.lock:
            if self.pending:
                self.log.append('console', level=self.label, message=self.pending)
                self.pending = ''
            self.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def training_log(path, run_id=None):
    log = EventLog(path, run_id)
    stdout = _Transcript(sys.stdout, log, 'INFO')
    stderr = _Transcript(sys.stderr, log, 'ERROR')
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            yield log
        except BaseException as error:
            log.append('run_failed', level='ERROR', error_type=type(error).__name__,
                       message=str(error), traceback=traceback.format_exc())
            print('Training run stopped with an error or interruption.', file=sys.stderr)
            traceback.print_exc()
            raise
        finally:
            stdout.finish()
            stderr.finish()
