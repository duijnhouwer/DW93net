"""Interactive CLI and safe process control for training and evaluation."""
import json
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from run_revision import run_directory_name


def _registry():
    # Same per-user location as run_lock; independent of checkout and TMPDIR.
    return Path('/tmp') / f'dw93net-run-locks-{os.getuid()}'


@contextmanager
def _control_lock():
    registry = _registry()
    registry.mkdir(mode=0o700, exist_ok=True)
    with (registry / 'control.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield registry / 'runner-control.json'


def _process_identity(pid):
    # Field 22 is the start time; split after the final ')' because comm may
    # contain spaces or parentheses. Boot ID prevents matching across reboots.
    stat = (Path('/proc') / str(pid) / 'stat').read_text()
    return {'start_ticks': stat.rsplit(')', 1)[1].split()[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def _read_active(path):
    try:
        record = json.loads(path.read_text())
        pid = record['pid']
        if (record.get('protocol') != 'dw93net-mission-v2' or type(pid) is not int
                or pid <= 0 or record.get('uid') != os.getuid()):
            raise RuntimeError('Unsupported runner control record; no command sent')
        if _process_identity(pid) != record.get('identity'):
            return None
        # A dead, unreaped child is not an active worker.
        stat = (Path('/proc') / str(pid) / 'stat').read_text()
        if stat.rsplit(')', 1)[1].split()[0] == 'Z':
            return None
        return record
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (KeyError, ValueError, TypeError) as error:
        raise RuntimeError('Invalid runner control record') from error


def active_runner():
    with _control_lock() as path:
        return _read_active(path)


def _send_command(action, expected_identity=None):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Mission control requires Linux pidfd support')
    signum = {'stop': signal.SIGUSR2, 'evaluate': signal.SIGUSR1}[action]
    with _control_lock() as path:
        record = _read_active(path)
        if record is None:
            raise RuntimeError('No ready training runner found')
        if expected_identity is not None and record['identity'] != expected_identity:
            raise RuntimeError('The active runner changed; refresh before sending a command')
        if not record.get('ready'):
            raise RuntimeError('Runner is starting; refresh shortly')
        if action == 'evaluate' and not Path(record['checkpoint']).is_file():
            raise RuntimeError('No completed checkpoint yet; evaluation is not available')
        if action == 'evaluate' and record['phase'] in ('final_test', 'duffy_wurtz_1993'):
            return record['pid']  # Already doing the requested work.
        pid = record['pid']
        try:
            descriptor = os.pidfd_open(pid)
            try:
                if _process_identity(pid) != record['identity']:
                    raise RuntimeError('Stale runner identity; no signal sent')
                status = (Path('/proc') / str(pid) / 'status').read_text()
                caught = next(line.split()[1] for line in status.splitlines()
                              if line.startswith('SigCgt:'))
                if not int(caught, 16) & (1 << (signum - 1)):
                    raise RuntimeError('Runner has no command handler; no signal sent')
                signal.pidfd_send_signal(descriptor, signum)
                _write_json(path, record | {'requested_action': action})
            finally:
                os.close(descriptor)
        except (OSError, StopIteration, ValueError) as error:
            raise RuntimeError('Runner is unavailable; command was not delivered') from error
    return pid


def request_finish_training():
    """Request test and Duffy-Wurtz evaluation using the selected saved weights."""
    return _send_command('evaluate')


def request_stop():
    """Stop at the next safe poll, without final evaluation."""
    return _send_command('stop')


def _write_json(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record) + '\n')
    os.replace(temporary, path)


class RunnerStopped(Exception):
    pass


class EvaluationRequested(Exception):
    pass


class RunnerControl:
    def __init__(self, log, mission_plan_path, checkpoint, mode='train'):
        self.log = log
        self.pending = mode == 'evaluate'
        self.stop_requested = False
        self.announced = False
        self.checkpoint = Path(checkpoint)
        self.record = dict(protocol='dw93net-mission-v2', pid=os.getpid(), uid=os.getuid(),
                           identity=_process_identity(os.getpid()), ready=True,
                           mission_plan=str(Path(mission_plan_path).resolve()),
                           checkpoint=str(self.checkpoint), phase='starting', mode=mode)

    def __enter__(self):
        with _control_lock() as path:
            self.previous = {signum: signal.getsignal(signum)
                             for signum in (signal.SIGUSR1, signal.SIGUSR2)}
            try:
                signal.signal(signal.SIGUSR1, self._evaluate)
                signal.signal(signal.SIGUSR2, self._stop)
                _write_json(path, self.record)
            except BaseException:
                for signum, handler in self.previous.items():
                    signal.signal(signum, handler)
                raise
        print(f'Mission control ready for {Path(self.record["mission_plan"]).stem}; PID {os.getpid()}', flush=True)
        return self

    def __exit__(self, *exc):
        with _control_lock() as path:
            try:
                path.unlink(missing_ok=True)
            finally:
                for signum, handler in self.previous.items():
                    signal.signal(signum, handler)

    def _evaluate(self, signum, frame):
        self.pending = True

    def _stop(self, signum, frame):
        self.stop_requested = True

    def set_phase(self, phase, **details):
        with _control_lock() as path:
            current = _read_active(path)
            self.record = self.record | details | {'phase': phase}
            if current and current.get('requested_action'):
                self.record['requested_action'] = current['requested_action']
            _write_json(path, self.record)

    def check_stop(self):
        if self.stop_requested:
            raise RunnerStopped('Stopped by mission control; latest completed checkpoint retained')

    def check_training(self):
        self.check_stop()
        if self() and self.checkpoint.is_file():
            raise EvaluationRequested('Evaluate saved checkpoint now')

    def __call__(self):
        self.check_stop()
        if self.pending and not self.announced:
            self.announced = True
            self.log.append('evaluation_requested')
            print('Evaluation requested: use saved weights; unfinished epoch is discarded. '
                  'If no checkpoint exists, complete the first epoch.', flush=True)
        return self.pending


def _lock_busy():
    registry = _registry()
    registry.mkdir(mode=0o700, exist_ok=True)
    with (registry / 'run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def _mission_plan_info(root, commit=None):
    import tomllib
    paths = sorted((root / 'scripts/mission_control/mission_plan').glob('*.toml'))
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one scripts/mission_control/mission_plan/*.toml file')
    mission_plan_path = paths[0]
    mission_plan = tomllib.loads(mission_plan_path.read_text())
    network = Path(mission_plan['network']['output_path']) / run_directory_name(mission_plan_path.stem, commit, root=root)
    return mission_plan_path, mission_plan, network


def repository_git_status(root):
    """Read checkout identity without importing the training runtime."""
    import subprocess

    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()

    try:
        commit = git('rev-parse', 'HEAD')
        branch = git('rev-parse', '--abbrev-ref', 'HEAD')
        changes = git('status', '--porcelain', '--untracked-files=all')
    except (OSError, subprocess.CalledProcessError):
        return dict(branch=None, commit=None, dirty=None)
    return dict(branch='detached HEAD' if branch == 'HEAD' else branch,
                commit=commit, dirty=bool(changes))


def show_git_status(root, network, output):
    current = repository_git_status(root)
    saved = None
    if network is not None:
        try:
            saved = json.loads((network / 'run.json').read_text())
            if not isinstance(saved, dict):
                raise ValueError('Invalid run manifest')
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            output('Saved run: metadata unavailable')
    if saved is not None:
        output(f'Saved run branch: {saved.get("git_branch") or "not recorded"}')
        output(f'Saved run commit: {saved.get("git_commit") or "not recorded"}')
    else:
        output('Saved run: no recorded branch/commit available')
    output(f'Current branch: {current["branch"] or "unavailable"}')
    output(f'Current commit: {current["commit"] or "unavailable"}')
    if current['dirty'] is None:
        output('Repository: UNKNOWN — cannot verify readiness to start/resume')
    elif current['dirty']:
        output('Repository: DIRTY — staged, unstaged, or untracked changes; start/resume blocked')
    else:
        output('Repository: CLEAN — no staged, unstaged, or untracked changes')
    if saved and current['commit'] and saved.get('git_commit') != current['commit']:
        output('Run revision: MISMATCH — resume/evaluation requires the saved run commit')


def launch_runner(mode='train', *, root=None):
    """Launch a detached runner; output goes to its mission-control log."""
    import subprocess
    import sys
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    if mode not in ('train', 'evaluate'):
        raise ValueError('mode must be train or evaluate')
    # Import only for a launch: refreshing the menu does not load Torch.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.mission_control.backend.run_mission_plan import load_mission_plan, _initialize_output
    from run_revision import committed_revision
    commit = committed_revision(root)
    mission_plan_path, _, network = _mission_plan_info(root, commit)
    mission_plan = load_mission_plan(mission_plan_path)
    if network.exists():
        manifest = json.loads((network / 'run.json').read_text())
        if any(manifest.get(key) != value for key, value in
               {'mission_plan': mission_plan, 'git_commit': commit}.items()):
            raise RuntimeError('Existing run mission plan/revision differs; its artifacts are preserved')
    checkpoint = network / 'checkpoints/latest.pt'
    if mode == 'evaluate' and not checkpoint.is_file():
        raise RuntimeError('No completed checkpoint to evaluate; start training first')
    output_root = Path(mission_plan['network']['output_path'])
    if not output_root.is_dir():
        raise RuntimeError('Configured output directory must already exist')
    log_path = network / 'mission-control.log'
    with _control_lock() as path:
        if _read_active(path) is not None or _lock_busy():
            raise RuntimeError('A runner or its generator is already active; refresh')
        # Publish the manifest before opening the transcript inside the run directory,
        # so the child can validate and reuse the directory at startup.
        _initialize_output(mission_plan, mission_plan_path.stem, commit, root=root,
                           git_branch=repository_git_status(root)['branch'])
        command = [sys.executable, '-u', str(root / 'scripts/mission_control/backend/run_mission_plan.py')]
        if mode == 'evaluate':
            command.append('--evaluate')
        with log_path.open('a') as output:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            _write_json(path, dict(protocol='dw93net-mission-v2', pid=child.pid, uid=os.getuid(),
                                  identity=_process_identity(child.pid), ready=False,
                                  mission_plan=str(mission_plan_path), checkpoint=str(checkpoint),
                                  phase='starting', mode=mode, log_path=str(log_path)))
        except BaseException:
            child.terminate()
            child.wait(timeout=10)
            raise
    return child


def show_training_log(network, output=print):
    """Launch the report script with the same Python environment as the menu."""
    import subprocess
    import sys

    if network is None:
        raise RuntimeError('No run output directory available for the training report')
    network = Path(network).resolve()
    log_path = network / 'training_log.jsonl'
    if not log_path.is_file():
        output(f'No training log found: {log_path}')
        return
    script = Path(__file__).parent / 'backend' / 'report_training_log.py'
    result = subprocess.run([sys.executable, str(script), str(log_path),
                             '--run-id', str(network / 'checkpoints'),
                             '--dashboard-gui'])
    if result.returncode:
        raise RuntimeError(f'Training report exited with status {result.returncode}')


def mission_control(*, root=None, input_fn=input, output=print):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    children = []
    try:
        while True:
            for child in children[:]:
                if child.poll() is not None:
                    if child.returncode:
                        output(f'Runner exited with status {child.returncode}; check its mission-control log.')
                    children.remove(child)
            try:
                current = active_runner()
                if current:
                    name = Path(current['mission_plan']).stem
                    phase = current['phase']
                    request = current.get('requested_action')
                    suffix = f' ({request} requested)' if request else ''
                    epoch = f'; epoch {current["global_epoch"]}' if current.get('global_epoch') else ''
                    output(f'\nMission plan: {name}\nStatus: {phase}{epoch}{suffix}; PID {current["pid"]}')
                    option = 'Stop'
                    checkpoint = current.get('checkpoint')
                    network = Path(checkpoint).parent.parent if checkpoint else None
                else:
                    mission_plan_path, mission_plan, network = _mission_plan_info(root)
                    busy = _lock_busy()
                    option = 'Resume training' if (network / 'run.json').exists() else 'Start'
                    status = ('Runner/generator is active but not controllable; wait for cleanup or use its original tools'
                              if busy else 'Stopped / idle' if option == 'Resume training' else 'Not started')
                    output(f'\nMission plan: {mission_plan_path.stem}\nStatus: {status}')
                    output(f'Run directory: {network}')
                show_git_status(root, network, output)
                output(f'1: Report training log\n2: {option}\n3: Evaluate now\n4: Refresh\nQ: Exit mission control')
                choice = input_fn('Choice: ').strip().upper()
                if choice == 'Q':
                    break
                if choice == '4':
                    continue
                try:
                    if choice == '1':
                        show_training_log(network, output)
                    elif choice == '2' and current:
                        _send_command('stop', current['identity'])
                        output('Stop requested. Resume training becomes available once cleanup finishes.')
                    elif choice in ('2', '3'):
                        if current:
                            _send_command('evaluate', current['identity'])
                            output('Evaluation requested.')
                        else:
                            children.append(launch_runner('train' if choice == '2' else 'evaluate', root=root))
                            output('Runner launched in the background.')
                    else:
                        output('Choose 1, 2, 3, 4, or Q.')
                except (RuntimeError, OSError, ValueError, KeyError, SystemExit) as error:
                    output(f'Cannot perform action: {error}')
            except (RuntimeError, OSError, ValueError, KeyError, SystemExit) as error:
                output(f'Cannot perform action: {error}')
                # On startup/status failure, still permit refresh or exit.
                if input_fn('Enter to refresh, or Q to exit: ').strip().upper() == 'Q':
                    break
    except (KeyboardInterrupt, EOFError):
        pass
    output('Mission control exited. Any background runner continues.')


if __name__ == '__main__':
    mission_control()
