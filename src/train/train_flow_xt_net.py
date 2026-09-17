"""Training engine called by the recipe orchestrator for one seeded set."""

import os
import math
import shutil
import time
from pathlib import Path
import torch
from dataclasses import dataclass, asdict

from models.pure_translation import PureTranslationNet
from features.reichardt import ReichardtBank
from stimuli.load_flow_xt_data import DoGDataset, make_flow_loader, with_mirrored_dog_training
from stimuli.metadata import require_pure_translation_root, motion_scales, data_contract
from checkpoints import SCHEMA, load_checkpoint
from precision import configure_tf32
from run_revision import committed_revision



@dataclass
class RecipeTrainingSettings:
    root_folder: str
    checkpoint_dir: str
    git_commit: str
    training_set_index: int
    training_seed: int
    epochs_per_set: int
    shuffle_seed: int
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


def train_recipe_set(cfg, *, progress_callback=None):
    """Train one seeded dataset, resuming only at completed epoch boundaries."""
    require_pure_translation_root(cfg.root_folder)
    if type(cfg.training_set_index) is not int or cfg.training_set_index < 0:
        raise ValueError('training_set_index must be a nonnegative integer')
    for name in ('epochs_per_set', 'candidate_interval_epochs', 'batch_size'):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configure_tf32(cfg.allow_tp32)
    train_cache = DoGDataset(Path(cfg.root_folder) / 'train')
    correlation_bank = ReichardtBank(cfg.dx_values)
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
    model = PureTranslationNet(correlation_bank.dx_values).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    amp_enabled = device == 'cuda' and cfg.allow_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.learning_rate_factor,
        patience=cfg.learning_rate_patience, min_lr=cfg.learning_rate_minimum,
        threshold=cfg.learning_rate_threshold, threshold_mode='rel')

    checkpoint_dir = Path(cfg.checkpoint_dir)
    latest_path = checkpoint_dir / 'latest.pt'
    candidates_dir = checkpoint_dir / 'candidates'
    start_epoch = 0
    if latest_path.exists():
        checkpoint = load_checkpoint(latest_path, device, resume=True, expected_commit=cfg.git_commit)
        if checkpoint['data_contract'] != contract:
            raise ValueError('Checkpoint dataset geometry or generation code does not match')
        if checkpoint.get('detector') != detector_metadata:
            raise ValueError('Checkpoint detector configuration does not match cache')
        previous_index = checkpoint['training_set_index']
        previous_epoch = checkpoint['epoch_within_set']
        if previous_index == cfg.training_set_index:
            if checkpoint['training_seed'] != cfg.training_seed:
                raise ValueError('Checkpoint training seed does not match the recipe')
            start_epoch = previous_epoch
        elif previous_index == cfg.training_set_index - 1:
            if previous_epoch != cfg.epochs_per_set:
                raise ValueError('Previous training set checkpoint is incomplete')
        else:
            raise ValueError('Checkpoint training-set position does not match the recipe')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'Resumed set {cfg.training_set_index + 1} seed {cfg.training_seed} '
              f'after epoch {start_epoch}', flush=True)

    for epoch_index in range(start_epoch, cfg.epochs_per_set):
        global_epoch = cfg.training_set_index * cfg.epochs_per_set + epoch_index + 1
        shuffle_generator.manual_seed(cfg.shuffle_seed + global_epoch)
        model.train()
        train_loss_total = torch.zeros((), device=device)
        epoch_start = time.perf_counter()
        data_wait = 0.0
        gpu_events = []
        iterator = iter(train_loader)
        window_start = time.perf_counter()
        window_wait = 0.0
        for batch_index in range(len(train_loader)):
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
            raise RuntimeError(
                f'Nonfinite training loss {epoch_loss}: set {cfg.training_set_index + 1}, '
                f'seed {cfg.training_seed}, epoch {epoch_index + 1}, global epoch {global_epoch}; '
                'previous checkpoint retained')
        previous_rate = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_loss)
        next_rate = optimizer.param_groups[0]['lr']
        elapsed = time.perf_counter() - epoch_start
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
        if next_rate < previous_rate:
            print(f'Training-loss plateau: learning rate reduced to {next_rate:.6g}', flush=True)

        common = {
            'allow_tp32': cfg.allow_tp32,
            'allow_amp': cfg.allow_amp,
            'schema': SCHEMA, 'data_contract': contract,
            'git_commit': cfg.git_commit,
            'global_epoch': global_epoch,
            'training_set_index': cfg.training_set_index,
            'training_seed': cfg.training_seed,
            'epoch_within_set': epoch_index + 1,
            'epochs_per_set': cfg.epochs_per_set,
            'training_loss': epoch_loss,
            'learning_rate': next_rate,
            'detector': detector_metadata,
            'n_input_channels': len(correlation_bank.dx_values),
            'model_state_dict': model.state_dict(),
        }
        checkpoint = common | {
            'kind': 'resume',
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'settings': asdict(cfg),
        }
        # Always offer the completed run to validation, even when its final
        # epoch falls between periodic candidates. Intermediate set boundaries
        # do not change the global candidate cadence.
        terminal_epoch = cfg.is_final_training_set and epoch_index + 1 == cfg.epochs_per_set
        if global_epoch % cfg.candidate_interval_epochs == 0 or terminal_epoch:
            candidate = common | {'kind': 'candidate'}
            _atomic_torch_save(candidate, candidates_dir / f'epoch-{global_epoch:06d}.pt')
        # Publish completion last. If candidate saving is interrupted, the epoch
        # is deliberately repeated rather than skipped on resume.
        _save_latest(checkpoint, latest_path, cfg.keep_previous_resumable)

    return latest_path
