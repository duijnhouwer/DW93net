"""Run the repository's single committed training recipe."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from parallel_stimuli import Prefetch
from run_lock import run_lock
from mission_control import RunnerControl, RunnerStopped, EvaluationRequested, repository_git_status

from checkpoints import load_checkpoint
from training_log import EventLog, training_log
from evaluation import inference_batches, summary_metrics, duffy_wurtz_summary, METRIC_DEFINITIONS
from stimuli.metadata import motion_scales, require_current_schema
from stimuli.preprocessing_provenance import capture_provenance
from stimuli.generate_flow_xt_data import StimConfig, generate_dog_dataset, validate_speed_float32
from stimuli.dataset_identity import dataset_spec, storage_manifest
from stimuli.load_flow_xt_data import DoGDataset
from train.train_flow_xt_net import (
    RecipeTrainingSettings, committed_revision, train_recipe_set, record_completed_epoch,
)

TOP_LEVEL_KEYS = {'version', 'require_cuda', 'network', 'stimuli', 'detector', 'training_sets',
                  'validation_set', 'test_set', 'optimization', 'learning_rate',
                  'checkpoints', 'evaluation', 'parallel_generation', 'duffy_wurtz_1993_set'}
STIMULI_KEYS = {'speed_range', 'heading_deg_range', 'rot_deg_range', 'fov_deg',
                 'near_exclusion_radius', 'n_world_dots', 'superimpose',
                 'x_supersampling', 'dog_bundle_size', 'apply_dog_filter_after_downsample'}
DETECTOR_KEYS = {'displacement_min_px', 'displacement_max_px', 'displacement_step_px',
                 'dog_sigma_center_px', 'dog_sigma_surround_px'}
TRAINING_KEYS = {'path', 'number_of_sets', 'samples_per_set', 'epochs_per_set',
                 'start_seed', 'seed_increment', 'shuffle_seed'}
FIXED_SET_KEYS = {'path', 'seed', 'n_samples'}
OPTIMIZATION_KEYS = {'batch_size', 'allow_amp', 'allow_tp32', 'profile_batches', 'num_workers',
                     'mirror_train_stimuli'}
LEARNING_RATE_KEYS = {'initial', 'scheduler', 'monitor', 'factor', 'patience_epochs',
                      'minimum', 'relative_improvement_threshold',
                      'reset_on_new_training_set'}
MANAGED_MARKER = '.dw93net-managed'
DATASET_STATE = '.dw93net-dataset.json'


def _reject_unknown(mapping, allowed, where):
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {where} field(s): {', '.join(sorted(unknown))}")


def _section(recipe, key, allowed):
    value = recipe.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'[{key}] is required')
    _reject_unknown(value, allowed, key)
    return value


def _positive_int(value, where):
    if type(value) is not int or value < 1:
        raise ValueError(f'{where} must be a positive integer')


def _nonnegative_int(value, where):
    if type(value) is not int or value < 0:
        raise ValueError(f'{where} must be a nonnegative integer')


def _number(value, where, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{where} must be a number')
    if not math.isfinite(value):
        raise ValueError(f'{where} must be finite')
    if positive and value <= 0:
        raise ValueError(f'{where} must be positive')
    if nonnegative and value < 0:
        raise ValueError(f'{where} must be nonnegative')


def _range(value, where):
    if (not isinstance(value, list) or len(value) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)):
        raise ValueError(f'{where} must contain exactly two numbers')
    if value[0] > value[1]:
        raise ValueError(f'{where} must be ordered from minimum to maximum')
    if not all(math.isfinite(x) for x in value):
        raise ValueError(f'{where} must contain finite numbers')


def _absolute_path(value, where):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f'{where} must be an absolute path')
    return Path(value).resolve()


def _validate_fixed_set(section, name):
    _nonnegative_int(section.get('seed'), f'{name}.seed')
    _positive_int(section.get('n_samples'), f'{name}.n_samples')
    return _absolute_path(section.get('path'), f'{name}.path')


def load_recipe(path):
    """Parse and strictly validate a recipe without changing the filesystem."""
    recipe = tomllib.loads(Path(path).read_text())
    _reject_unknown(recipe, TOP_LEVEL_KEYS, 'top-level')
    if recipe.get('version') != 1:
        raise ValueError('Recipe version must be 1')
    if type(recipe.get('require_cuda', True)) is not bool:
        raise ValueError('require_cuda must be true or false')
    network = _section(recipe, 'network', {'output_path'})
    output = _absolute_path(network.get('output_path'), 'network.output_path')

    stimuli = _section(recipe, 'stimuli', STIMULI_KEYS)
    for key in ('speed_range', 'heading_deg_range', 'rot_deg_range'):
        if key not in stimuli:
            raise ValueError(f'stimuli.{key} is required')
        _range(stimuli[key], f'stimuli.{key}')
    validate_speed_float32(stimuli['speed_range'])
    _positive_int(stimuli.get('dog_bundle_size'), 'stimuli.dog_bundle_size')
    for key in ('n_world_dots', 'x_supersampling'):
        if key in stimuli:
            _positive_int(stimuli[key], f'stimuli.{key}')
    if 'fov_deg' in stimuli:
        fov = stimuli['fov_deg']
        _number(fov, 'stimuli.fov_deg', positive=True)
        if fov > 360:
            raise ValueError('stimuli.fov_deg must not exceed 360')
    if 'near_exclusion_radius' in stimuli:
        radius = stimuli['near_exclusion_radius']
        _number(radius, 'stimuli.near_exclusion_radius', nonnegative=True)
        if radius >= 1:
            raise ValueError('stimuli.near_exclusion_radius must be less than 1')
    if 'superimpose' in stimuli and type(stimuli['superimpose']) is not bool:
        raise ValueError('stimuli.superimpose must be true or false')
    if type(stimuli.get('apply_dog_filter_after_downsample', True)) is not bool:
        raise ValueError('stimuli.apply_dog_filter_after_downsample must be true or false')

    StimConfig(**{key: value for key, value in stimuli.items()
                  if key in StimConfig.__dataclass_fields__})

    detector = _section(recipe, 'detector', DETECTOR_KEYS)
    for key in ('displacement_min_px', 'displacement_max_px'):
        if type(detector.get(key)) is not int:
            raise ValueError(f'detector.{key} must be an integer')
    _positive_int(detector.get('displacement_step_px'), 'detector.displacement_step_px')
    if detector['displacement_min_px'] > detector['displacement_max_px']:
        raise ValueError('detector displacement range must be ordered')
    dx_values = range(detector['displacement_min_px'], detector['displacement_max_px'] + 1,
                      detector['displacement_step_px'])
    dx_values = tuple(dx_values)
    if not dx_values or dx_values[-1] != detector['displacement_max_px']:
        raise ValueError('detector displacement step must land on displacement_max_px')
    if 0 not in dx_values or any(-value not in dx_values for value in dx_values):
        raise ValueError('detector displacements must contain zero and paired signs')
    for key in ('dog_sigma_center_px', 'dog_sigma_surround_px'):
        _number(detector.get(key), f'detector.{key}', positive=True)
    if detector['dog_sigma_center_px'] >= detector['dog_sigma_surround_px']:
        raise ValueError('detector DoG requires center sigma smaller than surround sigma')

    training = _section(recipe, 'training_sets', TRAINING_KEYS)
    for key in ('number_of_sets', 'samples_per_set', 'epochs_per_set', 'seed_increment'):
        _positive_int(training.get(key), f'training_sets.{key}')
    for key in ('start_seed', 'shuffle_seed'):
        _nonnegative_int(training.get(key), f'training_sets.{key}')
    training_path = _absolute_path(training.get('path'), 'training_sets.path')
    parallel = recipe.get('parallel_generation', {})
    if not isinstance(parallel, dict):
        raise ValueError('parallel_generation must be a table')
    _reject_unknown(parallel, {'enabled', 'min_threads', 'max_threads'}, 'parallel_generation')
    if type(parallel.get('enabled', False)) is not bool:
        raise ValueError('parallel_generation.enabled must be boolean')
    for key in ('min_threads', 'max_threads'):
        _positive_int(parallel.get(key, 1 if key == 'min_threads' else 8),
                      f'parallel_generation.{key}')
    if parallel.get('min_threads', 1) > parallel.get('max_threads', 8):
        raise ValueError('parallel_generation.min_threads must not exceed max_threads')

    validation = _section(recipe, 'validation_set', FIXED_SET_KEYS | {'selection_metric'})
    validation_path = _validate_fixed_set(validation, 'validation_set')
    if validation.get('selection_metric') != 'validation_loss':
        raise ValueError('validation_set.selection_metric must be "validation_loss"')

    test = _section(recipe, 'test_set', FIXED_SET_KEYS)
    test_path = _validate_fixed_set(test, 'test_set')
    duffy = _section(recipe, 'duffy_wurtz_1993_set', FIXED_SET_KEYS)
    duffy_path = _validate_fixed_set(duffy, 'duffy_wurtz_1993_set')
    seeds = [validation['seed'], test['seed'], duffy['seed']]
    if len(set(seeds)) != 3:
        raise ValueError('validation_set, test_set, and duffy_wurtz_1993_set seeds must differ')
    # Positive increments keep every training set above all held-out seeds,
    # regardless of how many sets are configured. No sequence enumeration needed.
    if training['start_seed'] <= max(seeds):
        raise ValueError('training_sets.start_seed must be greater than all '
                         'validation_set.seed, test_set.seed, and duffy_wurtz_1993_set.seed')

    optimization = _section(recipe, 'optimization', OPTIMIZATION_KEYS)
    _positive_int(optimization.get('batch_size'), 'optimization.batch_size')
    for key in ('profile_batches', 'num_workers'):
        _nonnegative_int(optimization.get(key), f'optimization.{key}')
    for key in ('allow_amp', 'allow_tp32', 'mirror_train_stimuli'):
        if type(optimization.get(key)) is not bool:
            raise ValueError(f'optimization.{key} must be true or false')

    lr = _section(recipe, 'learning_rate', LEARNING_RATE_KEYS)
    if lr.get('scheduler') != 'reduce_on_plateau' or lr.get('monitor') != 'validation_loss':
        raise ValueError('learning_rate must use reduce_on_plateau with validation_loss')
    for key in ('initial', 'factor', 'minimum'):
        _number(lr.get(key), f'learning_rate.{key}', positive=True)
    if not 0 < lr['factor'] < 1:
        raise ValueError('learning_rate.factor must be between 0 and 1')
    if lr['initial'] <= lr['minimum']:
        raise ValueError('learning_rate.initial must be greater than minimum')
    _nonnegative_int(lr.get('patience_epochs'), 'learning_rate.patience_epochs')
    _number(lr.get('relative_improvement_threshold'),
            'learning_rate.relative_improvement_threshold', nonnegative=True)
    if lr.get('reset_on_new_training_set') is not False:
        raise ValueError('learning_rate.reset_on_new_training_set must be false')

    checkpoints = _section(recipe, 'checkpoints',
                           {'candidate_interval_epochs', 'keep_previous_resumable'})
    _positive_int(checkpoints.get('candidate_interval_epochs'),
                  'checkpoints.candidate_interval_epochs')
    if type(checkpoints.get('keep_previous_resumable')) is not bool:
        raise ValueError('checkpoints.keep_previous_resumable must be true or false')

    evaluation = _section(recipe, 'evaluation', {'batch_size', 'num_workers', 'shuffle'})
    _positive_int(evaluation.get('batch_size'), 'evaluation.batch_size')
    _nonnegative_int(evaluation.get('num_workers'), 'evaluation.num_workers')
    if evaluation.get('shuffle') is not False:
        raise ValueError('evaluation.shuffle must be false')

    paths = {'network.output_path': output, 'training_sets.path': training_path,
             'validation_set.path': validation_path, 'test_set.path': test_path,
             'duffy_wurtz_1993_set.path': duffy_path}
    if parallel.get('enabled', False):
        paths['parallel_generation.staging'] = training_path.with_name(training_path.name + '.next')
    entries = list(paths.items())
    for i, (left_name, left) in enumerate(entries):
        for right_name, right in entries[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f'{left_name} and {right_name} must be separate, non-nested paths')
    return recipe


def discover_recipe(root=ROOT):
    """Find the sole directly-contained TOML recipe tracked in this commit."""
    root = Path(root).resolve()
    paths = sorted((root / 'recipe').glob('*.toml')) if (root / 'recipe').is_dir() else []
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one recipe/*.toml file; found {len(paths)}')
    path = paths[0]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', path.stem):
        raise ValueError('Recipe filename must be a filesystem-safe network name')
    try:
        subprocess.check_output(['git', '-C', str(root), 'ls-files', '--error-unmatch',
                                 '--', str(path.relative_to(root))], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise ValueError(f'Recipe must be tracked by Git: {path.relative_to(root)}') from error
    return path


def _initialize_output(recipe, network_name, commit, *, git_branch=None):
    output_root = Path(recipe['network']['output_path']).resolve()
    if not output_root.is_dir() or not os.access(output_root, os.W_OK):
        raise ValueError(f'output_path must already exist and be writable: {output_root}')
    network_dir = output_root / network_name
    expected = {'recipe': recipe, 'git_commit': commit}
    manifest = network_dir / 'run.json'
    if network_dir.exists():
        if not manifest.is_file():
            raise ValueError(f'Output has no current run manifest: {network_dir}')
        saved = json.loads(manifest.read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Output recipe or Git commit does not match: {network_dir}')
    else:
        network_dir.mkdir()
        _atomic_json(manifest, expected | {'git_branch': git_branch})
    checkpoint_dir, final_dir = network_dir / 'checkpoints', network_dir / 'final'
    checkpoint_dir.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True)
    return checkpoint_dir, final_dir


def _safe_delete_managed(path):
    path = Path(path).resolve()
    if not (path / MANAGED_MARKER).is_file():
        raise ValueError(f'Refusing to remove an unmanaged dataset directory: {path}')
    shutil.rmtree(path)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256_file(path, control_callback=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            if control_callback is not None:
                control_callback()
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sha256(cache_root, role, control_callback=None):
    """Hash the exact metadata, index, and tensor payloads used for evaluation."""
    cache_root = Path(cache_root).resolve()
    paths = [cache_root / 'dog.json', *sorted((cache_root / role).glob('*'))]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Invalid evaluation dataset file: {path}')
        relative = path.relative_to(cache_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, 'big'))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path, control_callback)))
    return digest.hexdigest()


def _cached_test_result(path, identity):
    """Return metrics only for a complete cache with an exact evaluation identity."""
    try:
        saved = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None
    if (not isinstance(saved, dict) or saved.get('version') != 1
            or saved.get('identity') != identity or not isinstance(saved.get('metrics'), dict)):
        return None
    try:
        _validate_metrics(saved['metrics'], 'cached test evaluation')
    except (ValueError, TypeError):
        return None
    return saved['metrics']


def _prepare_dataset(path, role, section, stimuli, *, persistent, progress_callback=None):
    """Generate one role-only set containing only bundled signed DoG contrast."""
    if section.get('_repo_root'):
        committed_revision(section['_repo_root'], section['_git_commit'])
    path = Path(path).resolve()
    desired = dataset_spec(role, section, stimuli)
    state_path = path / DATASET_STATE
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        cache = (path / state['cache']).resolve()
        # An obsolete cache requires explicit regeneration; never erase it as
        # a side effect of the exploratory schema change.
        metadata_path = cache / 'dog.json'
        if path in cache.parents and metadata_path.is_file():
            require_current_schema(json.loads(metadata_path.read_text()))
        ready = False
        if state.get('spec') == desired and path in cache.parents:
            try:
                ready = storage_manifest(cache, role, section['n_samples']) == state.get('storage')
            except (OSError, ValueError, TypeError, KeyError, EOFError) as error:
                print(f'Invalid {role} cache: {error}', flush=True)
        if state.get('spec') == desired and ready:
            print(f'Reusing {role} set: {path}', flush=True)
            return cache
        if persistent:
            raise ValueError(f'Persistent {role} set differs from the recipe or has invalid storage: {path}')
        _safe_delete_managed(path)
    elif path.exists() and any(path.iterdir()):
        # A managed directory with no published state is an interrupted
        # generation, not a persistent dataset. It is safe to start it over.
        if (path / MANAGED_MARKER).is_file():
            _safe_delete_managed(path)
        else:
            raise ValueError(f'Refusing to overwrite a nonempty unmanaged path: {path}')
    path.mkdir(parents=True, exist_ok=True)
    (path / MANAGED_MARKER).touch()
    config_values = {key: value for key, value in stimuli.items()
                     if key not in {'speed_range', 'heading_deg_range',
                                    'rot_deg_range', 'dog_bundle_size'}}
    detector = section['_detector']
    cache = generate_dog_dataset(
        output_root=str(path), dog_bundle_size=stimuli['dog_bundle_size'],
        sigma_center=detector['dog_sigma_center_px'],
        sigma_surround=detector['dog_sigma_surround_px'],
        n_samples=section['n_samples'], speed_range=stimuli['speed_range'],
        heading_deg_range=stimuli['heading_deg_range'], rot_deg_range=stimuli['rot_deg_range'],
        role=role, seed=section['seed'], config=StimConfig(**config_values),
        provenance=section.get('_provenance'),
        **({'progress_callback': progress_callback} if progress_callback is not None else {}))
    cache = Path(cache).resolve()
    if path not in cache.parents:
        raise RuntimeError('Generator returned a path outside its managed dataset directory')
    storage = storage_manifest(cache, role, section['n_samples'])
    _atomic_json(state_path, {'spec': desired, 'cache': str(cache.relative_to(path)),
                              'storage': storage})
    return cache


def _validate_metrics(metrics, where):
    """Undefined heading is legitimate; nonfinite numeric metrics are not."""
    _positive_int(metrics.get('n_samples'), f'{where}.n_samples')
    for key in ('normalized_rmse', 'translation_speed_mae',
                'translation_vector_tip_mae', 'rotation_mae_deg', 'heading_mae_deg'):
        if key == 'heading_mae_deg' and key in metrics and metrics[key] is None:
            continue
        _number(metrics.get(key), f'{where}.{key}', nonnegative=True)


def evaluate_model(model_path, cache_root, role, evaluation, commit, *, control_callback=None):
    """Evaluate a candidate/final model without shuffling or retaining predictions."""
    checkpoint = load_checkpoint(model_path, 'cpu', expected_commit=commit)
    return evaluate_checkpoint(checkpoint, cache_root, role, evaluation, commit,
                               **({"control_callback": control_callback} if control_callback else {}))


def evaluate_checkpoint(checkpoint, cache_root, role, evaluation, commit, *, control_callback=None):
    """All held-out evaluations use consistent full float32, even after fallback."""
    started = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = DoGDataset(Path(cache_root) / role)
    batches = inference_batches(checkpoint | {'allow_tp32': False}, dataset, device=device,
                                batch_size=evaluation['batch_size'],
                                num_workers=evaluation['num_workers'],
                                protocol='duffy_wurtz_1993' if role == 'duffy_wurtz_1993' else None,
                                control_callback=control_callback)
    summarize = duffy_wurtz_summary if role == 'duffy_wurtz_1993' else summary_metrics
    metrics = summarize(batches, motion_scales(cache_root))
    _validate_metrics(metrics, f'{role} evaluation of epoch {checkpoint["global_epoch"]}')
    return metrics | {'checkpoint_git_commit': checkpoint['git_commit'],
                      'evaluator_git_commit': commit, 'global_epoch': checkpoint['global_epoch'],
                      'evaluation_duration_s': time.perf_counter() - started,
                      'device': device, 'allow_tf32': False,
                      'dataset': {'path': str(Path(cache_root).resolve()), 'split': role,
                                  'generation': dataset.storage_metadata['stimulus_generation'],
                                  'selection': 'all'}, 'metric_definitions': METRIC_DEFINITIONS}


def _select_final(checkpoint_dir, final_dir, recipe, commit):
    """Promote only the best candidate committed by a healthy completed epoch."""
    log = EventLog(Path(recipe.get('network', {}).get('output_path', checkpoint_dir.parent))
                   / 'training_log.jsonl', str(checkpoint_dir.resolve()))
    latest = load_checkpoint(checkpoint_dir / 'latest.pt', resume=True, expected_commit=commit)
    name = latest['best_candidate']
    if not re.fullmatch(r'epoch-\d+\.pt', name):
        raise ValueError('Invalid best candidate filename')
    selected = load_checkpoint(checkpoint_dir / 'candidates' / name, expected_commit=commit)
    _number(selected['validation_loss'], 'best validation loss', nonnegative=True)
    if selected['validation_loss'] != latest['best_validation_loss']:
        raise ValueError('Best candidate does not match the resumable checkpoint')
    model_path = final_dir / 'model.pt'
    _atomic_torch_save(selected, model_path)
    print(f'Promoted {name} to final/model.pt using validation loss', flush=True)
    log.append('validation_completed', git_commit=commit, selection_metric='validation_loss',
               selected=name, selected_value=selected['validation_loss'], model_path=str(model_path))
    return True


def run_recipe(recipe_path=None, *, root=ROOT, commit=None, require_clean=True, mode='train'):
    if mode not in ('train', 'evaluate'):
        raise ValueError('mode must be train or evaluate')
    recipe_path = discover_recipe(root) if recipe_path is None else Path(recipe_path).resolve()
    recipe = load_recipe(recipe_path)
    with run_lock() as lock_fds:
        output_root = Path(recipe['network']['output_path'])
        if not output_root.is_dir() or not os.access(output_root, os.W_OK):
            raise ValueError(f'output_path must already exist and be writable: {output_root}')
        run_id = str((Path(recipe['network']['output_path']) / recipe_path.stem / 'checkpoints').resolve())
        with training_log(Path(recipe['network']['output_path']) / 'training_log.jsonl', run_id) as log:
            print(f'Starting/resuming training recipe: {recipe_path}', flush=True)
            if require_clean:
                commit = committed_revision(root, commit)
                # Re-read after checking Git so the run uses the verified recipe.
                if load_recipe(recipe_path) != recipe:
                    raise RuntimeError('Recipe changed while starting training')
            prefetch = Prefetch(recipe, lock_fds=lock_fds)
            prefetch.log_run_id = run_id
            log.append('run_started', recipe_path=str(recipe_path), git_commit=commit)
            with RunnerControl(log, recipe_path, Path(run_id) / 'latest.pt', mode) as finish:
                try:
                    result = _run_recipe(recipe_path, root=root, commit=commit,
                                         require_clean=require_clean, prefetch=prefetch,
                                         recipe=recipe, finish=finish)
                except RunnerStopped as error:
                    finish.set_phase('stopping')
                    log.append('run_stopped', git_commit=commit, reason='mission_control')
                    print(str(error), flush=True)
                    result = None
                finally:
                    prefetch.close()
            if not finish.stop_requested:
                log.append('run_completed', git_commit=commit)
            return result



def _run_recipe(recipe_path=None, *, root=ROOT, commit=None, require_clean=True, prefetch, recipe, finish):
    """Execute or resume the one committed recipe in ``root/recipe``."""
    root = Path(root).resolve()
    recipe_path = discover_recipe(root) if recipe_path is None else Path(recipe_path).resolve()
    if recipe_path.parent != root / 'recipe':
        raise ValueError('Recipe must be directly inside the repository recipe/ directory')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', recipe_path.stem):
        raise ValueError('Recipe filename must be a filesystem-safe network name')
    if recipe.get('require_cuda', True) and not torch.cuda.is_available():
        raise RuntimeError('Recipe requires CUDA, but torch.cuda.is_available() is false')
    if commit is None:
        commit = committed_revision(root) if require_clean else subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Git commit must be the full 40-character lowercase ID')

    def check_repository():
        if require_clean:
            committed_revision(root, commit)

    check_repository()
    run_fields = {'_git_commit': commit}
    if require_clean:
        run_fields['_repo_root'] = str(root)
    network_name = recipe_path.stem
    checkpoint_dir, final_dir = _initialize_output(recipe, network_name, commit,
                                                   git_branch=repository_git_status(root)['branch'])
    log = EventLog(Path(recipe['network']['output_path']) / 'training_log.jsonl',
                   str(checkpoint_dir.resolve()))
    finish.check_stop()
    if finish.record['mode'] == 'evaluate' and not (checkpoint_dir / 'latest.pt').is_file():
        raise ValueError('No completed checkpoint to evaluate; start training first')
    provenance = capture_provenance(checkpoint_dir.parent, ROOT)
    prefetch.provenance = provenance
    print(f'Network: {network_name}\nRecipe: {recipe_path}\nGit commit: {commit}', flush=True)

    stimuli, training = recipe['stimuli'], recipe['training_sets']
    validation = recipe['validation_set'] | run_fields
    validation = validation | {'_detector': recipe['detector'], '_provenance': provenance}
    validation_path = Path(validation['path'])
    finish.set_phase('preparing_validation')
    with prefetch.foreground_generation():
        validation_cache = _prepare_dataset(validation_path, 'validate', validation,
                                            stimuli, persistent=True, progress_callback=lambda count: finish())
    log.append('validation_set_ready', path=str(validation_cache), seed=validation['seed'],
               n_samples=validation['n_samples'])

    duffy = recipe['duffy_wurtz_1993_set'] | run_fields
    duffy = duffy | {'_detector': recipe['detector'], '_provenance': provenance}
    finish.set_phase('preparing_duffy_wurtz_1993')
    with prefetch.foreground_generation():
        duffy_cache = _prepare_dataset(Path(duffy['path']), 'duffy_wurtz_1993', duffy,
                                       stimuli | {'superimpose': True}, persistent=True,
                                       progress_callback=lambda count: finish())
    log.append('duffy_wurtz_1993_set_ready', path=str(duffy_cache), seed=duffy['seed'],
               n_samples=duffy['n_samples'], superimpose=True)

    def validate_epoch(checkpoint):
        check_repository()
        finish.set_phase('validation', global_epoch=checkpoint['global_epoch'])
        metrics = evaluate_checkpoint(checkpoint, validation_cache, 'validate', recipe['evaluation'], commit,
                                      control_callback=finish.check_training)
        check_repository()
        print(f'Validation performance: {json.dumps(metrics, sort_keys=True)}', flush=True)
        return metrics

    latest_path = checkpoint_dir / 'latest.pt'
    stop_reason = 'training_limit'
    for set_index in range(training['number_of_sets']):
        check_repository()
        seed = training['start_seed'] + set_index * training['seed_increment']
        complete = False
        if latest_path.is_file():
            latest = load_checkpoint(latest_path, 'cpu', resume=True, expected_commit=commit)
            record_completed_epoch(log, latest, recover=True)
            latest_index = latest['training_set_index']
            if not 0 <= latest_index < training['number_of_sets']:
                raise ValueError('Latest checkpoint has an invalid training-set index')
            expected_seed = (training['start_seed']
                             + latest_index * training['seed_increment'])
            if latest.get('training_seed') != expected_seed:
                raise ValueError('Latest checkpoint training seed does not match the recipe')
            if not 0 <= latest['epoch_within_set'] <= training['epochs_per_set']:
                raise ValueError('Latest checkpoint has an invalid epoch position')
            if latest['stop_reason'] == 'learning_rate_minimum' or finish():
                stop_reason = ('learning_rate_minimum' if latest['stop_reason'] == 'learning_rate_minimum'
                               else 'user_requested')
                break
            complete = (latest_index > set_index or (latest_index == set_index and
                        latest['epoch_within_set'] >= training['epochs_per_set']))
        training_path = Path(training['path'])
        if complete:
            log.append('training_set_skipped', training_set_index=set_index, training_seed=seed,
                       reason='already_completed')
            print(f'Training set {set_index + 1}/{training["number_of_sets"]} already complete',
                  flush=True)
            # The shared path may already contain a later set. Its manifest
            # is checked by _prepare_dataset when that set is reached.
            continue
        section = {'seed': seed, 'n_samples': training['samples_per_set'],
                   '_detector': recipe['detector'], '_provenance': provenance, **run_fields}
        print(f'Preparing training set {set_index + 1}/{training["number_of_sets"]}, seed {seed}',
              flush=True)
        log.append('training_set_preparing', training_set_index=set_index, training_seed=seed,
                   n_samples=training['samples_per_set'])
        try:
            finish.set_phase('preparing_training', training_set_index=set_index)
            prefetch.acquire(training_path, section, control_callback=finish.check_training)
            with prefetch.foreground_generation():
                cache = _prepare_dataset(training_path, 'train', section, stimuli, persistent=False,
                                         progress_callback=lambda count: finish.check_training())
            check_repository()
            if set_index + 1 < training['number_of_sets'] and not finish():
                prefetch.start(section | {'seed': seed + training['seed_increment']})
            optimization, lr, checkpoints = (recipe['optimization'], recipe['learning_rate'],
                                             recipe['checkpoints'])
            def training_progress(loader_wait):
                finish()
                if prefetch.enabled:
                    prefetch.observe(loader_wait)

            train_recipe_set(RecipeTrainingSettings(
                root_folder=str(cache), checkpoint_dir=str(checkpoint_dir), git_commit=commit,
                log_path=str(Path(recipe['network']['output_path']) / 'training_log.jsonl'),
                training_set_index=set_index, training_seed=seed,
                is_final_training_set=set_index == training['number_of_sets'] - 1,
                epochs_per_set=training['epochs_per_set'], shuffle_seed=training['shuffle_seed'],
                candidate_interval_epochs=checkpoints['candidate_interval_epochs'],
                keep_previous_resumable=checkpoints['keep_previous_resumable'],
                batch_size=optimization['batch_size'], allow_amp=optimization['allow_amp'],
                allow_tp32=optimization['allow_tp32'],
                profile_batches=optimization['profile_batches'], num_workers=optimization['num_workers'],
                learning_rate=lr['initial'], learning_rate_factor=lr['factor'],
                learning_rate_patience=lr['patience_epochs'], learning_rate_minimum=lr['minimum'],
                learning_rate_threshold=lr['relative_improvement_threshold'],
                mirror_train_stimuli=optimization['mirror_train_stimuli'],
                dx_values=tuple(range(recipe['detector']['displacement_min_px'],
                                      recipe['detector']['displacement_max_px'] + 1,
                                      recipe['detector']['displacement_step_px']))),
                validate_epoch=validate_epoch, should_finish=finish,
                control_callback=finish.check_training,
                epoch_callback=lambda epoch: finish.set_phase('training', global_epoch=epoch),
                progress_callback=training_progress)
        except EvaluationRequested:
            log.append('training_interrupted_for_evaluation', training_set_index=set_index)
            stop_reason = 'user_requested'
            break
        check_repository()
        latest = load_checkpoint(latest_path, resume=True, expected_commit=commit)
        if latest['stop_reason'] == 'learning_rate_minimum' or finish():
            stop_reason = ('learning_rate_minimum' if latest['stop_reason'] == 'learning_rate_minimum'
                           else 'user_requested')
            break
        _safe_delete_managed(training_path)
        print(f'Training set {set_index + 1} complete and deleted', flush=True)
        log.append('training_set_completed', training_set_index=set_index, training_seed=seed,
                   dataset_deleted=True)

    prefetch.close()
    log.append('training_stopped', reason=stop_reason)

    # A manual evaluation pause may leave epochs in the current set. Keep its
    # exact data for continuation; completed sets can use the normal cleanup.
    latest = load_checkpoint(latest_path, resume=True, expected_commit=commit)
    keep_current_set = (stop_reason == 'user_requested'
                        and latest['epoch_within_set'] < training['epochs_per_set'])
    training_path = Path(training['path'])
    if (not keep_current_set and training_path.exists()
            and (training_path / MANAGED_MARKER).is_file()):
        _safe_delete_managed(training_path)

    check_repository()
    finish.set_phase('selecting_best_checkpoint')
    finish.check_stop()
    _select_final(checkpoint_dir, final_dir, recipe, commit)

    check_repository()
    test = recipe['test_set'] | run_fields
    test = test | {'_detector': recipe['detector'], '_provenance': provenance}
    test_path = Path(test['path'])
    finish.set_phase('preparing_final_test')
    with prefetch.foreground_generation():
        test_cache = _prepare_dataset(test_path, 'test', test, stimuli, persistent=True,
                                      progress_callback=lambda count: finish())
    check_repository()
    model_path = final_dir / 'model.pt'
    for role, cache in (('test', test_cache), ('duffy_wurtz_1993', duffy_cache)):
        finish.check_stop()
        finish.set_phase('final_test' if role == 'test' else role)
        results_path = final_dir / f'{role}-results.json'
        identity = {'model_sha256': _sha256_file(model_path, finish.check_stop),
                    'dataset_sha256': _dataset_sha256(cache, role, finish.check_stop),
                    'evaluation': recipe['evaluation'], 'evaluator_git_commit': commit,
                    'split': role}
        check_repository()
        metrics = _cached_test_result(results_path, identity)
        reused = metrics is not None
        if reused:
            print(f'Reusing final {role} result', flush=True)
        else:
            print(f'Evaluating final model: {role}', flush=True)
            metrics = evaluate_model(model_path, cache, role, recipe['evaluation'], commit,
                                     control_callback=finish.check_stop)
            check_repository()
            _atomic_json(results_path, {'version': 1, 'identity': identity, 'metrics': metrics})
        log.append(f'{role}_result', git_commit=commit, cached=reused,
                   model_sha256=identity['model_sha256'],
                   dataset_sha256=identity['dataset_sha256'],
                   model_path=str(final_dir / 'model.pt'), metrics=metrics)
        label = 'Test' if role == 'test' else 'Duffy-Wurtz 1993'
        print(f'{label} performance: {json.dumps(metrics, sort_keys=True)}', flush=True)
    check_repository()
    finish.check_stop()
    finish.set_phase('completed')
    print('Training recipe complete.', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the best saved checkpoint without training')
    args = parser.parse_args()
    run_recipe(mode='evaluate' if args.evaluate else 'train')


if __name__ == '__main__':
    main()
