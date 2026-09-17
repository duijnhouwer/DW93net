"""Self-contained Git-visible working source snapshot for a preprocessing run."""
import importlib.metadata
import io
import json
import platform
import subprocess
import tarfile
import os
import tempfile
from pathlib import Path


def capture_provenance(run_directory, repo):
    """Publish one source/environment snapshot per run, referenced by datasets."""
    run_directory = Path(run_directory).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    destination = run_directory / 'provenance'
    if destination.exists():
        return json.loads((destination / 'snapshot.json').read_text())
    with tempfile.TemporaryDirectory(prefix='.provenance-', dir=run_directory) as temporary:
        info, initial = write_provenance(temporary, repo, exclude=run_directory)
        if repository_state(repo, run_directory) != initial:
            raise RuntimeError('Source changed while capturing run provenance')
        info['directory'] = str(destination)
        (Path(temporary) / 'provenance' / 'snapshot.json').write_text(
            json.dumps(info, indent=2, allow_nan=False) + '\n')
        os.replace(Path(temporary) / 'provenance', destination)
    return info


def repository_state(repo, exclude):
    repo, exclude = Path(repo).resolve(), Path(exclude).resolve()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])

    commit = git('rev-parse', 'HEAD').decode().strip()
    diff = git('diff', '--binary', '--no-ext-diff', '--no-textconv', 'HEAD', '--')
    names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0')
    files = {}
    for raw in sorted(set(names)):
        if not raw:
            continue
        name = raw.decode('utf-8', errors='surrogateescape')
        path = repo / name
        if path == exclude or exclude in path.parents:
            continue
        if path.is_symlink():
            raise ValueError(f"Source snapshot requires ordinary files, found symlink: {name}")
        if path.is_file():
            content = path.read_bytes()
            files[name] = (content, path.stat().st_mode & 0o777)
    return commit, diff, files


def write_provenance(destination, repo, *, exclude=None):
    """Capture before processing. Return state for an end-of-run stability check."""
    destination = Path(destination)
    state = repository_state(repo, destination if exclude is None else exclude)
    commit, diff, files = state
    out = destination / 'provenance'
    out.mkdir()
    (out / 'working-tree.patch').write_bytes(diff)
    # Unlike a patch, this also includes untracked source and requires no Git
    # object database to recover the current code. No stimuli/.git/environment copy.
    with tarfile.open(out / 'source.tar.gz', 'w:gz') as archive:
        for name, (content, mode) in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), mode
            archive.addfile(info, io.BytesIO(content))
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       packages={dist.metadata['Name']: dist.version
                                 for dist in importlib.metadata.distributions() if dist.metadata['Name']})
    (out / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    (out / 'RESTORE.md').write_text(
        '# Recover this run\n\nExtract `source.tar.gz` into an empty directory to recover '
        'the working source directly, including Git-visible untracked files. '
        'Do not overlay an existing checkout: files deleted at capture time are absent.\n\n'
        f'Base Git commit: `{commit}`. `working-tree.patch` records tracked staged and '
        'unstaged changes relative to HEAD (binary changes included); it does not contain '
        'untracked files, which are in the archive. '
        '`environment.json` records installed package versions, not bundled dependencies.\n\n'
        'Scope: tracked and nonignored untracked repository files, excluding this output '
        'directory. Git-ignored datasets, virtual environments and other ignored files are '
        'not bundled. Git history is not bundled.\n')
    return dict(git_commit=commit, tracked_changes=bool(diff),
                directory='provenance',
                snapshot_scope='tracked and nonignored untracked working-tree files'), state
