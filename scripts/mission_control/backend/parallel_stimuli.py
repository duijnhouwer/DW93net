"""One-set lookahead with a separately spawned, CPU-only generator."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


class Throttle:
    """Two pressure windows to decrease; four quiet windows to increase."""

    def __init__(self, minimum, maximum):
        self.minimum, self.maximum = minimum, maximum
        self.threads = minimum
        self.high = self.low = 0

    def observe(self, wait_fraction):
        self.high = self.high + 1 if wait_fraction > 0.15 else 0
        self.low = self.low + 1 if wait_fraction < 0.03 else 0
        if self.high >= 2:
            self.threads = max(self.minimum, self.threads // 2)
            self.high = self.low = 0
        elif self.low >= 4:
            self.threads = min(self.maximum, self.threads + 1)
            self.high = self.low = 0
        return self.threads


class Prefetch:
    def __init__(self, mission_plan, *, lock_fds=()):
        self.mission_plan = mission_plan
        self.lock_fds = lock_fds
        self.provenance = None
        self.log_run_id = None
        self.settings = mission_plan.get('parallel_generation', {})
        self.enabled = self.settings.get('enabled', False)
        self.stage = Path(mission_plan['training_sets']['path']).with_name(
            Path(mission_plan['training_sets']['path']).name + '.next')
        self.process = self.temporary = None
        self.owns_stage = False
        cpus = (sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                else list(range(os.cpu_count() or 1)))
        reserve = min(max(1, mission_plan['optimization']['num_workers']), max(1, len(cpus) // 2))
        self.cpus = cpus[reserve:] or cpus
        maximum = min(self.settings.get('max_threads', 8), len(self.cpus) or 1)
        self.throttle = Throttle(min(self.settings.get('min_threads', 1), maximum), maximum)

    @contextmanager
    def foreground_generation(self):
        """Use the shared maximum while training is idle, then restore its threads."""
        import torch
        previous = torch.get_num_threads()
        threads = self.throttle.maximum
        torch.set_num_threads(threads)
        print(f'Foreground stimulus generation: {threads} CPU threads '
              f'(configured range {self.throttle.minimum}..{self.throttle.maximum})', flush=True)
        try:
            yield
        finally:
            torch.set_num_threads(previous)
            print(f'Restored {previous} CPU threads after generation', flush=True)

    def start(self, section):
        if not self.enabled:
            return
        if self.process is not None:
            raise RuntimeError('A generation job is already running')
        if section.get('_repo_root'):
            from run_revision import committed_revision
            committed_revision(section['_repo_root'], section['_git_commit'])
        section = section | {'_provenance': self.provenance}
        self.owns_stage = True
        self.temporary = tempfile.TemporaryDirectory(prefix='dw93net-prefetch-')
        self.control = Path(self.temporary.name) / 'control.json'
        self.request = Path(self.temporary.name) / 'request.json'
        self._write_control(self.throttle.threads)
        self.request.write_text(json.dumps(dict(
            path=str(self.stage), section=section, stimuli=self.mission_plan['stimuli'],
            log_run_id=self.log_run_id,
            log_path=(str(Path(self.log_run_id).parent / 'training_log.jsonl')
                      if self.log_run_id else None),
            cpus=self.cpus, control=str(self.control), parent_pid=os.getpid())))
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = '1'
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve()), str(self.request)], env=env,
            pass_fds=self.lock_fds)
        print(f'Generating next set in {self.stage}; seed {section["seed"]}; '
              f'{self.throttle.threads} threads (maximum {self.throttle.maximum})', flush=True)

    def _write_control(self, threads):
        temporary = self.control.with_suffix('.tmp')
        temporary.write_text(json.dumps({'threads': threads}))
        os.replace(temporary, self.control)

    def observe(self, wait_fraction):
        if self.process is None:
            return
        status = self.process.poll()
        if status is not None:
            if status:
                raise RuntimeError(f'Background stimulus generation failed: exit {status}')
            return
        previous = self.throttle.threads
        threads = self.throttle.observe(wait_fraction)
        if previous != threads:
            self._write_control(threads)
            print(f'Generator allocation: {threads} threads; '
                  f'loader wait {wait_fraction:.1%}', flush=True)

    def acquire(self, destination, section, control_callback=None):
        from stimuli.dataset_identity import dataset_spec

        if self.process is None:
            state_path = self.stage / '.dw93net-dataset.json'
            if not self.enabled or not state_path.is_file() or Path(destination).exists():
                return False
            state = json.loads(state_path.read_text())
            desired = dataset_spec('train', section, self.mission_plan['stimuli'])
            if state.get('spec') != desired:
                return False
            self.stage.rename(destination)
            print('Recovered completed staged training set', flush=True)
            return True
        start = time.perf_counter()
        if self.process.poll() is None:
            self._write_control(self.throttle.maximum)
        while True:
            if control_callback is not None:
                control_callback()
            try:
                status = self.process.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
        if status:
            raise RuntimeError(f'Background stimulus generation failed: exit {status}')
        # The child publishes dataset state last; never promote partial output.
        if not (self.stage / '.dw93net-dataset.json').is_file():
            raise RuntimeError('Background generator did not publish a complete dataset')
        if Path(destination).exists():
            raise RuntimeError(f'Cannot promote over existing training data: {destination}')
        self.stage.rename(destination)
        print(f'Next set ready; boundary wait {time.perf_counter() - start:.2f}s', flush=True)
        self.process = None
        self.temporary.cleanup()
        self.temporary = None
        return True

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.temporary is not None:
            self.temporary.cleanup()
        # Keep completed lookahead for a restart; incomplete owned data is disposable.
        if self.owns_stage and (self.stage / '.dw93net-managed').is_file() and not (
                self.stage / '.dw93net-dataset.json').is_file():
            import shutil
            shutil.rmtree(self.stage)


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    from contextlib import nullcontext
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
    from training_log import training_log
    with (training_log(request['log_path'], request.get('log_run_id'))
          if request.get('log_path') else nullcontext()):
        print(f'Background generation starting: seed {request["section"]["seed"]}', flush=True)
        _worker(request)
        print('Background generation complete', flush=True)


def _worker(request):
    if hasattr(os, 'nice'):
        os.nice(10)
    if request['cpus'] and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, request['cpus'])
    # Best effort: I/O priorities depend on the host's scheduler and permissions.
    try:
        subprocess.run(['ionice', '-c', '2', '-n', '7', '-p', str(os.getpid())],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        print('Generator I/O priority unavailable; CPU priority remains lowered', flush=True)
    import torch
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.mission_control.backend.run_mission_plan import _prepare_dataset
    current = None

    def progress(count):
        nonlocal current
        if count % 64 and count != 0:
            return
        if os.getppid() != request['parent_pid']:
            raise RuntimeError('Training parent exited; stopping background generation')
        threads = json.loads(Path(request['control']).read_text())['threads']
        if threads != current:
            torch.set_num_threads(threads)
            current = threads
            if request['cpus'] and hasattr(os, 'sched_setaffinity'):
                # Update existing numerical-library threads as well as the main thread.
                for task in Path('/proc/self/task').iterdir():
                    try:
                        os.sched_setaffinity(int(task.name), request['cpus'][:threads])
                    except ProcessLookupError:
                        pass
            print(f'Generator applied {threads} CPU threads at sample {count}', flush=True)

    progress(0)
    _prepare_dataset(request['path'], 'train', request['section'], request['stimuli'],
                     persistent=False, progress_callback=progress)


if __name__ == '__main__':
    worker(sys.argv[1])
