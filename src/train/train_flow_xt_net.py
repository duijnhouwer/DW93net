"""Training engine called by the mission plan orchestrator for one seeded set."""

import os
import copy
import json
import math
import shutil
import time
from pathlib import Path
import torch
from dataclasses import dataclass, asdict

from models.flatland_flow import FlatlandFlowNet
from features.reichardt import ReichardtBank
from stimuli.load_flow_xt_data import DoGDataset, make_flow_loader, with_mirrored_dog_training
from stimuli.metadata import motion_scales, data_contract
from checkpoints import SCHEMA, load_checkpoint, validate_tensor_health, validate_checkpoint, NonfiniteTensorError
from precision import configure_tf32
from run_revision import committed_revision
from training_log import EventLog
from evaluation import NonfiniteEvaluationError



class NumericalTrainingError(RuntimeError):
    """A nonfinite loss or state eligible for one full-float32 retry."""


@dataclass
class MissionPlanTrainingSettings:
    root_folder: str
    checkpoint_dir: str
    git_commit: str
    training_set_index: int
    training_seed: int
    epochs_per_set: int
    shuffle_seed: int
    log_path: str | None = None
    is_final_training_set: bool = True
    candidate_interval_epochs: int = 1
    keep_previous_resumable: bool = True
    batch_size: int = 64
    allow_amp: bool = True
    allow_tp32: bool = True
    profile_batches: int = 20
    num_workers: int = 8
    learning_rate: float = 1e-3
    learning_rate_factor: float = 0.5
    learning_rate_patience: int = 10
    learning_rate_minimum: float = 1e-6
    learning_rate_threshold: float = 1e-4
    mirror_train_stimuli: bool = True
    dx_values: tuple = tuple(range(-60, 61))


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def _save_latest(checkpoint, path, keep_previous):
    path = Path(path)
    if keep_previous and path.exists():
        previous = path.with_name('previous.pt')
        temporary = previous.with_name(previous.name + '.tmp')
        shutil.copy2(path, temporary)
        os.replace(temporary, previous)
    _atomic_torch_save(checkpoint, path)


def _normalized_motion_loss(prediction, target, scales):
    return sum(torch.nn.functional.mse_loss(prediction[:, index], target[:, index]) / scale**2
               for index, scale in enumerate(scales))


def record_completed_epoch(log, checkpoint, *, recover=False, timing=None):
    """Repair a log interruption after atomic checkpoint publication on restart."""
    epoch = checkpoint['global_epoch']
    recorded = log.events() if recover else []
    candidate_name = f'epoch-{epoch:06d}.pt'
    if not any(e['event'] == 'validation_result' and e['result'].get('global_epoch') == epoch
               for e in recorded):
        log.append('validation_result', result=checkpoint['validation'] | {
            'candidate': candidate_name, 'global_epoch': epoch, 'validation_loss': checkpoint['validation_loss']},
            selected_as_best=checkpoint['best_candidate'] == candidate_name,
            next_learning_rate=checkpoint['learning_rate'], recovered_from_checkpoint=recover)
    if not any(e['event'] == 'epoch_completed' and e.get('global_epoch') == epoch for e in recorded):
        log.append('epoch_completed', **checkpoint['epoch_record'], **(timing or {}),
                   recovered_from_checkpoint=recover)


def train_mission_plan_set(cfg, *, validate_epoch, progress_callback=None, should_finish=None,
                     control_callback=None, epoch_callback=None):
    """Train one seeded dataset, resuming only at completed epoch boundaries."""
    if type(cfg.training_set_index) is not int or cfg.training_set_index < 0:
        raise ValueError('training_set_index must be a nonnegative integer')
    for name in ('epochs_per_set', 'candidate_interval_epochs', 'batch_size'):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = Path(cfg.checkpoint_dir)
    latest_path = checkpoint_dir / 'latest.pt'
    candidates_dir = checkpoint_dir / 'candidates'
    log = EventLog(cfg.log_path or checkpoint_dir.parent / 'training_log.jsonl',
                   str(checkpoint_dir.resolve()))
    allow_amp, allow_tp32 = cfg.allow_amp, cfg.allow_tp32
    fallbacks = log.events('precision_fallback')
    if fallbacks:
        fallback = fallbacks[-1]
        if fallback.get('git_commit') != cfg.git_commit:
            raise ValueError('Precision fallback Git commit does not match the run')
        allow_amp = allow_tp32 = False
        print(f'Persistent precision fallback loaded from {log.path}: AMP and TF32 OFF',
              flush=True)
    configure_tf32(allow_tp32)
    train_cache = DoGDataset(Path(cfg.root_folder) / 'train')
    correlation_bank = ReichardtBank(cfg.dx_values, boundary=train_cache.metadata['dog']['boundary'])
    detector_metadata = {'dog': train_cache.metadata,
                         'correlation': correlation_bank.metadata}
    train_dataset = train_cache
    if cfg.mirror_train_stimuli:
        train_dataset = with_mirrored_dog_training(train_dataset)
        print(f'Mirrored training augmentation: {len(train_dataset):,} presentations per epoch')
    if not len(train_dataset):
        raise ValueError('Training dataset must be nonempty')
    scales = motion_scales(cfg.root_folder)
    contract = data_contract(train_cache.storage_metadata)
    shuffle_generator = torch.Generator()
    worker_generator = torch.Generator().manual_seed(cfg.shuffle_seed)
    train_loader = make_flow_loader(
        train_dataset, cfg.batch_size, device, shuffle=True,
        num_workers=cfg.num_workers, shuffle_generator=shuffle_generator,
        worker_generator=worker_generator)

    torch.manual_seed(cfg.shuffle_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.shuffle_seed)
    model = FlatlandFlowNet(correlation_bank.dx_values, boundary=correlation_bank.boundary).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    amp_enabled = device == 'cuda' and allow_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.learning_rate_factor,
        patience=cfg.learning_rate_patience, min_lr=cfg.learning_rate_minimum,
        threshold=cfg.learning_rate_threshold, threshold_mode='rel', eps=0)

    start_epoch = 0
    best_validation_loss = math.inf
    best_candidate = None
    if latest_path.exists():
        checkpoint = load_checkpoint(latest_path, device, resume=True, expected_commit=cfg.git_commit)
        record_completed_epoch(log, checkpoint, recover=True)
        if checkpoint['data_contract'] != contract:
            raise ValueError('Checkpoint dataset geometry or generation code does not match')
        if checkpoint.get('detector') != detector_metadata:
            raise ValueError('Checkpoint detector configuration does not match cache')
        previous_index = checkpoint['training_set_index']
        previous_epoch = checkpoint['epoch_within_set']
        if previous_index == cfg.training_set_index:
            if checkpoint['training_seed'] != cfg.training_seed:
                raise ValueError('Checkpoint training seed does not match the mission plan')
            start_epoch = previous_epoch
        elif previous_index == cfg.training_set_index - 1:
            if previous_epoch != cfg.epochs_per_set:
                raise ValueError('Previous training set checkpoint is incomplete')
        else:
            raise ValueError('Checkpoint training-set position does not match the mission plan')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        allow_amp = allow_amp and checkpoint['allow_amp']
        allow_tp32 = allow_tp32 and checkpoint['allow_tp32']
        amp_enabled = device == 'cuda' and allow_amp
        configure_tf32(allow_tp32)
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        if amp_enabled:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_validation_loss = checkpoint['best_validation_loss']
        best_candidate = checkpoint['best_candidate']
        if checkpoint['stop_reason'] == 'learning_rate_minimum':
            return latest_path
        print(f'Resumed set {cfg.training_set_index + 1} seed {cfg.training_seed} '
              f'after epoch {start_epoch}', flush=True)

    print(f'Training precision: device={device}; AMP={amp_enabled}; TF32={allow_tp32}',
          flush=True)
    log.append('training_precision', device=device, allow_amp=allow_amp,
               amp_enabled=amp_enabled, allow_tf32=allow_tp32, git_commit=cfg.git_commit)
    # Epoch one has no checkpoint yet; retain its pristine initialization.
    initial_state = None
    if not latest_path.exists():
        initial_state = copy.deepcopy({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        })

    for epoch_index in range(start_epoch, cfg.epochs_per_set):
        if should_finish is not None and should_finish() and latest_path.exists():
            break
        global_epoch = cfg.training_set_index * cfg.epochs_per_set + epoch_index + 1
        if epoch_callback is not None:
            epoch_callback(global_epoch)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else None
        worker_rng = worker_generator.get_state()
        epoch_wall_start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            epoch_start = time.perf_counter()
            log.append('epoch_started', global_epoch=global_epoch, attempt=attempt,
                       training_set_index=cfg.training_set_index, training_seed=cfg.training_seed,
                       allow_amp=allow_amp, allow_tf32=allow_tp32)
            try:
                shuffle_generator.manual_seed(cfg.shuffle_seed + global_epoch)
                model.train()
                train_loss_total = torch.zeros((), device=device)
                data_wait = 0.0
                gpu_events = []
                iterator = iter(train_loader)
                window_start = time.perf_counter()
                window_wait = 0.0
                for batch_index in range(len(train_loader)):
                    if control_callback is not None:
                        control_callback()
                    wait_start = time.perf_counter()
                    x, y = next(iterator)
                    batch_wait = time.perf_counter() - wait_start
                    data_wait += batch_wait
                    window_wait += batch_wait
                    events = None
                    if device == 'cuda' and batch_index < cfg.profile_batches:
                        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                        events[0].record()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    x = correlation_bank(x[:, None], validate=False)
                    if events:
                        events[1].record()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                        prediction = model(x)
                    loss = _normalized_motion_loss(prediction.float(), y, scales)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss_total += loss.detach() * x.size(0)
                    if events:
                        events[2].record()
                        gpu_events.append(events)
                    if progress_callback is not None and (batch_index + 1) % 32 == 0:
                        now = time.perf_counter()
                        progress_callback(window_wait / max(now - window_start, 1e-9))
                        window_start, window_wait = now, 0.0

                epoch_loss = train_loss_total.item() / len(train_dataset)
                # Do not advance the scheduler or publish a checkpoint for a failed
                # epoch. Resume must retain the last completed, finite-loss epoch.
                if not math.isfinite(epoch_loss):
                    raise NumericalTrainingError(
                        f'Nonfinite training loss {epoch_loss}: set {cfg.training_set_index + 1}, '
                        f'seed {cfg.training_seed}, epoch {epoch_index + 1}, global epoch {global_epoch}; '
                        'previous checkpoint retained')
                def check_health(check):
                    try:
                        check()
                    except NonfiniteTensorError as error:
                        raise NumericalTrainingError(
                            f'{error}: set {cfg.training_set_index + 1}, '
                            f'seed {cfg.training_seed}, epoch {epoch_index + 1}, global epoch {global_epoch}; '
                            'previous checkpoint retained') from error

                # A finite pre-update loss can accompany an overflowing Adam moment or
                # a parameter corrupted by the final update. Check before advancing
                # the scheduler, and validate the complete checkpoint before publishing.
                check_health(lambda: validate_tensor_health({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }))
                elapsed = time.perf_counter() - epoch_start
                previous_rate = optimizer.param_groups[0]['lr']
                print(f'Set {cfg.training_set_index + 1} seed {cfg.training_seed}; '
                      f'epoch {epoch_index + 1}/{cfg.epochs_per_set}; global epoch {global_epoch}; '
                      f'loss {epoch_loss:.6f}; learning rate {previous_rate:.6g}; '
                      f'{len(train_dataset) / elapsed:.1f} samples/s; loader wait {data_wait:.2f}s',
                      flush=True)
                if gpu_events:
                    transfer_ms = sum(a.elapsed_time(b) for a, b, c in gpu_events) / len(gpu_events)
                    compute_ms = sum(b.elapsed_time(c) for a, b, c in gpu_events) / len(gpu_events)
                    print(f'GPU sampled {len(gpu_events)} batches: transfer/correlation '
                          f'{transfer_ms:.2f}ms; '
                          f'forward/backward/optimizer {compute_ms:.2f}ms per batch', flush=True)
                common = {
                    'allow_tp32': allow_tp32,
                    'allow_amp': allow_amp,
                    'schema': SCHEMA, 'data_contract': contract,
                    'git_commit': cfg.git_commit,
                    'global_epoch': global_epoch,
                    'training_set_index': cfg.training_set_index,
                    'training_seed': cfg.training_seed,
                    'epoch_within_set': epoch_index + 1,
                    'epochs_per_set': cfg.epochs_per_set,
                    'training_loss': epoch_loss,
                    'learning_rate': previous_rate,
                    'detector': detector_metadata,
                    'n_input_channels': len(correlation_bank.dx_values),
                    'model_state_dict': model.state_dict(),
                }
                validation_start = time.perf_counter()
                try:
                    validation = validate_epoch(common | {'kind': 'candidate'})
                finally:
                    configure_tf32(allow_tp32)
                validation_loss = validation['normalized_mse']
                if not math.isfinite(validation_loss):
                    raise NumericalTrainingError(f'Nonfinite validation loss at global epoch {global_epoch}')
                if validation_loss < 0:
                    raise ValueError('Validation loss must be nonnegative')
                validation_duration = time.perf_counter() - validation_start
                scheduler.step(validation_loss)
                next_rate = optimizer.param_groups[0]['lr']
                improved = validation_loss < best_validation_loss
                candidate_name = f'epoch-{global_epoch:06d}.pt'
                finish_requested = should_finish is not None and should_finish()
                stop_reason = ('learning_rate_minimum' if next_rate <= cfg.learning_rate_minimum
                               else 'user_requested' if finish_requested else None)
                common.update(validation=validation, validation_loss=validation_loss, learning_rate=next_rate)
                print(f'Validation loss {validation_loss:.6g}; next learning rate {next_rate:.6g}', flush=True)
                if next_rate < previous_rate:
                    print(f'Validation-loss plateau: learning rate reduced to {next_rate:.6g}', flush=True)
                checkpoint = common | {
                    'kind': 'resume',
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'settings': asdict(cfg) | {'allow_amp': allow_amp, 'allow_tp32': allow_tp32},
                    'best_validation_loss': validation_loss if improved else best_validation_loss,
                    'best_candidate': candidate_name if improved else best_candidate,
                    'stop_reason': stop_reason,
                }
                check_health(lambda: validate_checkpoint(checkpoint, resume=True, expected_commit=cfg.git_commit))
                break
            except (NumericalTrainingError, NonfiniteEvaluationError) as error:
                failure = {
                    'attempt': attempt,
                    'attempt_duration_s': time.perf_counter() - epoch_start,
                    'git_commit': cfg.git_commit,
                    'training_set_index': cfg.training_set_index,
                    'training_seed': cfg.training_seed,
                    'epoch_within_set': epoch_index + 1,
                    'global_epoch': global_epoch,
                    'error': str(error),
                    'allow_amp': allow_amp,
                    'allow_tp32': allow_tp32,
                }
                log.append('numerical_failure', **failure)
                print(f'Numerical failure details: {json.dumps(failure, sort_keys=True)}', flush=True)
                if not (allow_amp or allow_tp32):
                    print('Full float32 failed; stopping for investigation.', flush=True)
                    raise NumericalTrainingError(
                        f'{error}; full float32 failed, stopping for investigation') from error
                # Publish the sticky policy before retrying, so an interruption
                # cannot cause a restart to re-enable the failing precision mode.
                log.append('precision_fallback', **failure,
                           action='retry_epoch_in_full_float32',
                           new_allow_amp=False, new_allow_tf32=False)
                print(f'Numerical failure: {error}\n'
                      'Restoring the last healthy epoch and retrying with AMP and TF32 OFF. '
                      'Full float32 will remain enabled for the rest of this run.', flush=True)
                healthy = (load_checkpoint(latest_path, device, resume=True,
                                          expected_commit=cfg.git_commit)
                           if latest_path.exists() else copy.deepcopy(initial_state))
                model.load_state_dict(healthy['model_state_dict'])
                optimizer.load_state_dict(healthy['optimizer_state_dict'])
                scheduler.load_state_dict(healthy['scheduler_state_dict'])
                optimizer.zero_grad(set_to_none=True)
                allow_amp = allow_tp32 = amp_enabled = False
                configure_tf32(False)
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                worker_generator.set_state(worker_rng)
                # The next attempt reseeds the sampler to the same epoch/order.
        # Retain every new validation best, periodic snapshots, and the terminal
        # epoch. The latest checkpoint commits which candidate is actually best.
        if control_callback is not None:
            control_callback()
        checkpoint_start = time.perf_counter()
        checkpoint['epoch_record'] = dict(git_commit=cfg.git_commit,
            training_set_index=cfg.training_set_index, training_seed=cfg.training_seed,
            epoch_within_set=epoch_index + 1, global_epoch=global_epoch,
            training_loss=epoch_loss, learning_rate=next_rate, learning_rate_used=previous_rate,
            attempt=attempt, training_duration_s=elapsed, validation_duration_s=validation_duration,
            validation_loss=validation_loss, loader_wait_s=data_wait,
            samples_per_second=len(train_dataset) / elapsed, presentations=len(train_dataset),
            batches=len(train_loader), allow_amp=allow_amp, allow_tf32=allow_tp32,
            checkpoint=str(latest_path))
        terminal_epoch = cfg.is_final_training_set and epoch_index + 1 == cfg.epochs_per_set
        if improved or global_epoch % cfg.candidate_interval_epochs == 0 or terminal_epoch or stop_reason:
            candidate = common | {'kind': 'candidate'}
            _atomic_torch_save(candidate, candidates_dir / f'epoch-{global_epoch:06d}.pt')
        # Publish completion last. If candidate saving is interrupted, the epoch
        # is deliberately repeated rather than skipped on resume.
        _save_latest(checkpoint, latest_path, cfg.keep_previous_resumable)
        best_validation_loss = checkpoint['best_validation_loss']
        best_candidate = checkpoint['best_candidate']
        record_completed_epoch(log, checkpoint, timing={
            'epoch_wall_duration_s': time.perf_counter() - epoch_wall_start,
            'checkpoint_duration_s': time.perf_counter() - checkpoint_start})
        print(f'Healthy checkpoint saved: {latest_path}; global epoch {global_epoch}', flush=True)
        if stop_reason or (should_finish is not None and should_finish()):
            break

    return latest_path
