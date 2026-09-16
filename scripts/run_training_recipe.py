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
import tomllib

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from parallel_stimuli import Prefetch

from models.pure_translation import PureTranslationNet
from features.reichardt import ReichardtBank
from stimuli.generate_flow_xt_data import StimConfig, generate_dog_dataset
from stimuli.load_flow_xt_data import DoGDataset, make_flow_loader
from train.train_flow_xt_net import (
    RecipeTrainingSettings, _motion_scales, committed_revision, train_recipe_set,
)

TOP_LEVEL_KEYS = {'version', 'require_cuda', 'network', 'stimuli', 'detector', 'training_sets',
                  'validation_set', 'test_set', 'optimization', 'learning_rate',
                  'checkpoints', 'evaluation', 'parallel_generation'}
STIMULI_KEYS = {'speed_range', 'heading_deg_range', 'rot_deg_range', 'fov_deg',
                 'near_exclusion_radius', 'n_world_dots', 'superimpose',
                 'x_supersampling', 'dog_bundle_size'}
DETECTOR_KEYS = {'displacement_min_px', 'displacement_max_px', 'displacement_step_px',
                 'dog_sigma_center_px', 'dog_sigma_surround_px'}
TRAINING_KEYS = {'path', 'number_of_sets', 'samples_per_set', 'epochs_per_set',
                 'start_seed', 'seed_increment', 'shuffle_seed'}
FIXED_SET_KEYS = {'path', 'seed', 'n_samples'}
OPTIMIZATION_KEYS = {'batch_size', 'mixed_precision', 'profile_batches', 'num_workers',
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
    if stimuli['rot_deg_range'] != [0, 0]:
        raise ValueError('stimuli.rot_deg_range must be [0, 0] while rotation is disabled')
    _positive_int(stimuli.get('dog_bundle_size'), 'stimuli.dog_bundle_size')
    for key in ('n_world_dots', 'x_supersampling'):
        if key in stimuli:
            _positive_int(stimuli[key], f'stimuli.{key}')
    for key in ('fov_deg', 'near_exclusion_radius'):
        if key in stimuli:
            _number(stimuli[key], f'stimuli.{key}', positive=True)
    if 'superimpose' in stimuli and type(stimuli['superimpose']) is not bool:
        raise ValueError('stimuli.superimpose must be true or false')

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

    validation = _section(recipe, 'validation_set', FIXED_SET_KEYS |
                          {'keep_after_run', 'selection_metric'})
    validation_path = _validate_fixed_set(validation, 'validation_set')
    if validation.get('selection_metric') != 'normalized_rmse':
        raise ValueError('validation_set.selection_metric must be "normalized_rmse"')
    if type(validation.get('keep_after_run', False)) is not bool:
        raise ValueError('validation_set.keep_after_run must be true or false')

    test = _section(recipe, 'test_set', FIXED_SET_KEYS | {'staging_path'})
    test_path = _validate_fixed_set(test, 'test_set')

    optimization = _section(recipe, 'optimization', OPTIMIZATION_KEYS)
    _positive_int(optimization.get('batch_size'), 'optimization.batch_size')
    for key in ('profile_batches', 'num_workers'):
        _nonnegative_int(optimization.get(key), f'optimization.{key}')
    for key in ('mixed_precision', 'mirror_train_stimuli'):
        if type(optimization.get(key)) is not bool:
            raise ValueError(f'optimization.{key} must be true or false')

    lr = _section(recipe, 'learning_rate', LEARNING_RATE_KEYS)
    if lr.get('scheduler') != 'reduce_on_plateau' or lr.get('monitor') != 'training_loss':
        raise ValueError('learning_rate must use reduce_on_plateau with training_loss')
    for key in ('initial', 'factor', 'minimum'):
        _number(lr.get(key), f'learning_rate.{key}', positive=True)
    if not 0 < lr['factor'] < 1:
        raise ValueError('learning_rate.factor must be between 0 and 1')
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
             'validation_set.path': validation_path, 'test_set.path': test_path}
    if 'staging_path' in test:
        paths['test_set.staging_path'] = _absolute_path(test['staging_path'],
                                                        'test_set.staging_path')
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


def _initialize_output(recipe, network_name, commit):
    output_root = Path(recipe['network']['output_path']).resolve()
    if not output_root.is_dir() or not os.access(output_root, os.W_OK):
        raise ValueError(f'output_path must already exist and be writable: {output_root}')
    network_dir = output_root / network_name
    expected_marker = f'commit_id={commit}'
    if network_dir.exists():
        markers = list(network_dir.glob('commit_id=*'))
        if len(markers) != 1 or markers[0].name != expected_marker:
            raise ValueError(f'Output does not belong to Git commit {commit}: {network_dir}')
        if markers[0].stat().st_size:
            raise ValueError(f'Commit marker must be empty: {markers[0]}')
    else:
        network_dir.mkdir()
        (network_dir / expected_marker).touch()
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def _atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256(path):
    """Return a stable identity for a checkpoint that may be replaced on resume."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_dataset(path, role, section, stimuli, *, persistent, progress_callback=None):
    """Generate one role-only set containing only bundled signed DoG contrast."""
    path = Path(path).resolve()
    desired = {'version': 2, 'role': role, 'seed': section['seed'],
               'n_samples': section['n_samples'], 'stimuli': stimuli,
               'detector': section['_detector']}
    state_path = path / DATASET_STATE
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        cache = path / state['cache']
        ready = ((cache / 'dog.json').is_file()
                 and (cache / role / 'bundles.json').is_file())
        if state.get('spec') == desired and ready:
            print(f'Reusing {role} set: {path}', flush=True)
            return cache
        if persistent:
            raise ValueError(f'Persistent {role} set differs from the recipe: {path}')
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
        **({'progress_callback': progress_callback} if progress_callback is not None else {}))
    cache = Path(cache).resolve()
    if path not in cache.parents:
        raise RuntimeError('Generator returned a path outside its managed dataset directory')
    _atomic_json(state_path, {'spec': desired, 'cache': str(cache.relative_to(path))})
    return cache


def _load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if 'model_state_dict' not in checkpoint:
        raise ValueError(f'Checkpoint has no model weights: {path}')
    return checkpoint


def evaluate_model(model_path, cache_root, role, evaluation, commit):
    """Evaluate a candidate/final model without shuffling or retaining predictions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = DoGDataset(Path(cache_root) / role)
    loader = make_flow_loader(dataset, evaluation['batch_size'], device, shuffle=False,
                              num_workers=evaluation['num_workers'])
    checkpoint = _load_checkpoint(model_path, device)
    if checkpoint.get('git_commit') != commit:
        raise ValueError(f'Model Git commit does not match this run: {model_path}')
    detector = checkpoint.get('detector')
    if not isinstance(detector, dict) or detector.get('dog') != dataset.metadata:
        raise ValueError(f'Model detector configuration does not match {role} data')
    correlation_bank = ReichardtBank(detector['correlation']['dx_values'])
    if correlation_bank.metadata != detector['correlation']:
        raise ValueError('Model correlation configuration is unsupported')
    model = PureTranslationNet(correlation_bank.dx_values).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scales = torch.tensor(_motion_scales(cache_root), device=device)
    normalized_sq = speed_error = tip_error = rotation_error = heading_error = 0.0
    heading_count = count = 0
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            prediction = model(correlation_bank(inputs[:, None], validate=False)).float()
            difference = prediction - target
            normalized_sq += ((difference / scales) ** 2).sum().item()
            target_speed = torch.linalg.vector_norm(target[:, :2], dim=1)
            predicted_speed = torch.linalg.vector_norm(prediction[:, :2], dim=1)
            speed_error += (predicted_speed - target_speed).abs().sum().item()
            tip_error += torch.linalg.vector_norm(difference[:, :2], dim=1).sum().item()
            rotation_error += difference[:, 2].abs().sum().item()
            valid = (target_speed > 1e-12) & (predicted_speed > 1e-12)
            if valid.any():
                actual = torch.atan2(target[valid, 1], target[valid, 0])
                predicted = torch.atan2(prediction[valid, 1], prediction[valid, 0])
                angle = torch.atan2(torch.sin(predicted - actual), torch.cos(predicted - actual))
                heading_error += torch.rad2deg(angle.abs()).sum().item()
                heading_count += valid.sum().item()
            count += target.shape[0]
    if not count:
        raise ValueError(f'{role} dataset is empty')
    return {'n_samples': count, 'normalized_rmse': math.sqrt(normalized_sq / count),
            'translation_speed_mae': speed_error / count,
            'translation_vector_tip_mae': tip_error / count,
            'rotation_mae_deg': rotation_error / count,
            'heading_mae_deg': heading_error / heading_count if heading_count else None}


def _select_final(checkpoint_dir, final_dir, validation_cache, recipe, commit):
    candidates = sorted((checkpoint_dir / 'candidates').glob('epoch-*.pt'))
    if not candidates:
        raise ValueError('No candidate model checkpoints were saved')

    # Validation can take much longer than training when every epoch is a
    # candidate. Publish each result independently so an interruption resumes
    # at the first unevaluated (or subsequently replaced) checkpoint.
    progress_path = final_dir / 'validation-progress.json'
    context = {'version': 1, 'git_commit': commit,
               'selection_metric': 'normalized_rmse',
               'evaluation': recipe['evaluation']}
    progress = {**context, 'complete': False, 'candidates': []}
    if progress_path.is_file():
        saved = json.loads(progress_path.read_text())
        if any(saved.get(key) != value for key, value in context.items()):
            raise ValueError('Saved validation progress does not match this recipe run')
        if not isinstance(saved.get('candidates'), list):
            raise ValueError('Saved validation progress has invalid candidate results')
        progress = saved
    saved_by_name = {}
    for result in progress['candidates']:
        if (not isinstance(result, dict) or
                not isinstance(result.get('candidate'), str) or
                not isinstance(result.get('checkpoint_sha256'), str)):
            raise ValueError('Saved validation progress has an invalid candidate result')
        if result['candidate'] in saved_by_name:
            raise ValueError('Saved validation progress has duplicate candidate results')
        saved_by_name[result['candidate']] = result

    results = []
    for index, candidate in enumerate(candidates, 1):
        fingerprint = _sha256(candidate)
        saved = saved_by_name.get(candidate.name)
        if saved is not None and saved['checkpoint_sha256'] == fingerprint:
            print(f'Validation already complete {index}/{len(candidates)}: {candidate.name}',
                  flush=True)
            result = saved
        else:
            print(f'Validating candidate {index}/{len(candidates)}: {candidate.name}', flush=True)
            metrics = evaluate_model(candidate, validation_cache, 'validate',
                                     recipe['evaluation'], commit)
            result = {'candidate': candidate.name,
                      'checkpoint_sha256': fingerprint, **metrics}
            saved_by_name[candidate.name] = result
            # Preserve candidate order and any results for later candidates;
            # they remain reusable if a previously evaluated file changed.
            progress['candidates'] = sorted(
                saved_by_name.values(), key=lambda item: item['candidate'])
            progress['complete'] = False
            progress.pop('selected', None)
            progress.pop('selected_value', None)
            _atomic_json(progress_path, progress)
        results.append(result)
    best = min(results, key=lambda item: item['normalized_rmse'])
    _atomic_json(final_dir / 'validation-results.json', {
        'git_commit': commit, 'selection_metric': 'normalized_rmse',
        'selected': best['candidate'], 'selected_value': best['normalized_rmse'],
        'candidates': [{key: value for key, value in result.items()
                        if key != 'checkpoint_sha256'} for result in results]})
    progress['candidates'] = results
    progress['complete'] = True
    progress['selected'] = best['candidate']
    progress['selected_value'] = best['normalized_rmse']
    _atomic_json(progress_path, progress)
    model_path = final_dir / 'model.pt'
    promote = True
    if model_path.is_file():
        existing = _load_checkpoint(model_path, 'cpu')
        existing_value = existing.get('validation', {}).get('normalized_rmse')
        promote = (existing.get('git_commit') != commit or existing_value is None or
                   best['normalized_rmse'] < existing_value)
    if promote:
        selected = _load_checkpoint(checkpoint_dir / 'candidates' / best['candidate'], 'cpu')
        selected['validation'] = {'candidate': best['candidate'],
                                  'normalized_rmse': best['normalized_rmse']}
        _atomic_torch_save(selected, model_path)
        print(f'Promoted {best["candidate"]} to final/model.pt', flush=True)
    else:
        print('Existing final model remains the best validation result', flush=True)
    return promote


def _stage_test_set(test_path, staging_path):
    staging_path = Path(staging_path).resolve()
    if staging_path.exists():
        if (staging_path / MANAGED_MARKER).is_file():
            _safe_delete_managed(staging_path)
        elif any(staging_path.iterdir()):
            raise ValueError(f'Refusing to overwrite nonempty test staging path: {staging_path}')
        else:
            staging_path.rmdir()
    print(f'Copying test set to staging path: {staging_path}', flush=True)
    shutil.copytree(test_path, staging_path)
    (staging_path / MANAGED_MARKER).touch()
    state = json.loads((staging_path / DATASET_STATE).read_text())
    return staging_path / state['cache']


def run_recipe(recipe_path=None, *, root=ROOT, commit=None, require_clean=True):
    recipe_path = discover_recipe(root) if recipe_path is None else Path(recipe_path).resolve()
    prefetch = Prefetch(load_recipe(recipe_path))
    try:
        return _run_recipe(recipe_path, root=root, commit=commit,
                           require_clean=require_clean, prefetch=prefetch)
    finally:
        prefetch.close()


def _run_recipe(recipe_path=None, *, root=ROOT, commit=None, require_clean=True, prefetch):
    """Execute or resume the one committed recipe in ``root/recipe``."""
    root = Path(root).resolve()
    recipe_path = discover_recipe(root) if recipe_path is None else Path(recipe_path).resolve()
    recipe = load_recipe(recipe_path)
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
    network_name = recipe_path.stem
    checkpoint_dir, final_dir = _initialize_output(recipe, network_name, commit)
    print(f'Network: {network_name}\nRecipe: {recipe_path}\nGit commit: {commit}', flush=True)

    stimuli, training = recipe['stimuli'], recipe['training_sets']
    latest_path = checkpoint_dir / 'latest.pt'
    for set_index in range(training['number_of_sets']):
        seed = training['start_seed'] + set_index * training['seed_increment']
        complete = False
        if latest_path.is_file():
            latest = _load_checkpoint(latest_path, 'cpu')
            if latest.get('git_commit') != commit:
                raise ValueError('Latest checkpoint Git commit does not match this run')
            latest_index = latest['training_set_index']
            if not 0 <= latest_index < training['number_of_sets']:
                raise ValueError('Latest checkpoint has an invalid training-set index')
            expected_seed = (training['start_seed']
                             + latest_index * training['seed_increment'])
            if latest.get('training_seed') != expected_seed:
                raise ValueError('Latest checkpoint training seed does not match the recipe')
            if not 0 <= latest['epoch_within_set'] <= training['epochs_per_set']:
                raise ValueError('Latest checkpoint has an invalid epoch position')
            complete = (latest_index > set_index or (latest_index == set_index and
                        latest['epoch_within_set'] >= training['epochs_per_set']))
        training_path = Path(training['path'])
        if complete:
            print(f'Training set {set_index + 1}/{training["number_of_sets"]} already complete',
                  flush=True)
            if training_path.exists() and (training_path / MANAGED_MARKER).is_file():
                _safe_delete_managed(training_path)
            continue
        section = {'seed': seed, 'n_samples': training['samples_per_set'],
                   '_detector': recipe['detector']}
        print(f'Preparing training set {set_index + 1}/{training["number_of_sets"]}, seed {seed}',
              flush=True)
        prefetch.acquire(training_path, section)
        cache = _prepare_dataset(training_path, 'train', section, stimuli, persistent=False)
        if set_index + 1 < training['number_of_sets']:
            prefetch.start(section | {'seed': seed + training['seed_increment']})
        optimization, lr, checkpoints = (recipe['optimization'], recipe['learning_rate'],
                                         recipe['checkpoints'])
        train_recipe_set(RecipeTrainingSettings(
            root_folder=str(cache), checkpoint_dir=str(checkpoint_dir), git_commit=commit,
            training_set_index=set_index, training_seed=seed,
            epochs_per_set=training['epochs_per_set'], shuffle_seed=training['shuffle_seed'],
            candidate_interval_epochs=checkpoints['candidate_interval_epochs'],
            keep_previous_resumable=checkpoints['keep_previous_resumable'],
            batch_size=optimization['batch_size'], mixed_precision=optimization['mixed_precision'],
            profile_batches=optimization['profile_batches'], num_workers=optimization['num_workers'],
            learning_rate=lr['initial'], learning_rate_factor=lr['factor'],
            learning_rate_patience=lr['patience_epochs'], learning_rate_minimum=lr['minimum'],
            learning_rate_threshold=lr['relative_improvement_threshold'],
            mirror_train_stimuli=optimization['mirror_train_stimuli'],
            dx_values=tuple(range(recipe['detector']['displacement_min_px'],
                                  recipe['detector']['displacement_max_px'] + 1,
                                  recipe['detector']['displacement_step_px']))),
            **({'progress_callback': prefetch.observe} if prefetch.enabled else {}))
        _safe_delete_managed(training_path)
        print(f'Training set {set_index + 1} complete and deleted', flush=True)

    validation = recipe['validation_set']
    validation = validation | {'_detector': recipe['detector']}
    validation_path = Path(validation['path'])
    validation_cache = _prepare_dataset(validation_path, 'validate', validation,
                                        stimuli, persistent=False)
    promoted = _select_final(checkpoint_dir, final_dir, validation_cache, recipe, commit)
    if not validation.get('keep_after_run', False):
        _safe_delete_managed(validation_path)
        print('Validation set deleted after model selection', flush=True)

    test = recipe['test_set']
    test = test | {'_detector': recipe['detector']}
    test_path = Path(test['path'])
    test_cache = _prepare_dataset(test_path, 'test', test, stimuli, persistent=True)
    staged = bool(test.get('staging_path'))
    if staged:
        test_cache = _stage_test_set(test_path, test['staging_path'])
    results_path = final_dir / 'test-results.json'
    if promoted or not results_path.is_file():
        print('Testing final model', flush=True)
        metrics = evaluate_model(final_dir / 'model.pt', test_cache, 'test',
                                 recipe['evaluation'], commit)
        _atomic_json(results_path, {'git_commit': commit, **metrics})
    else:
        print('Final model is unchanged; retaining its existing test results', flush=True)
    if staged:
        _safe_delete_managed(Path(test['staging_path']))
        print('Staged test copy deleted', flush=True)
    print('Training recipe complete.', flush=True)


def main():
    if len(sys.argv) != 1:
        raise SystemExit('This command takes no arguments; it runs the sole recipe/*.toml file')
    run_recipe()


if __name__ == '__main__':
    main()
