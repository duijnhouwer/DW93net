"""Checkpoint structure and exact Git-commit matching."""

import math
import torch

from run_revision import revision

SCHEMA = 'dw93net-checkpoint-v4'


class NonfiniteTensorError(ValueError):
    """Numerical corruption, distinct from schema or run-identity errors."""


def validate_tensor_health(value, path='checkpoint'):
    """Reject NaN/Inf in nested floating/complex tensors, naming the bad state.

    Scalar scheduler sentinels such as mode_worse=inf are legitimate; this
    checks tensor payloads rather than rejecting every infinite Python number.
    """
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise NonfiniteTensorError(f'Nonfinite tensor in {path}')
    elif isinstance(value, dict):
        for key, child in value.items():
            validate_tensor_health(child, f'{path}[{key!r}]')
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_tensor_health(child, f'{path}[{index}]')


def validate_checkpoint(checkpoint, *, resume=False, expected_commit=None):
    common = {'schema', 'kind', 'data_contract', 'git_commit', 'allow_amp', 'allow_tp32',
              'global_epoch', 'training_set_index', 'training_seed', 'epoch_within_set',
              'epochs_per_set', 'training_loss', 'learning_rate', 'detector',
              'n_input_channels', 'model_state_dict'}
    if not isinstance(checkpoint, dict) or common - checkpoint.keys():
        raise ValueError('Incomplete checkpoint schema; legacy checkpoints are not supported')
    if checkpoint['schema'] != SCHEMA or checkpoint['kind'] not in ('candidate', 'resume'):
        raise ValueError('Unsupported checkpoint schema or kind; regenerate checkpoints')
    for key in ('allow_amp', 'allow_tp32'):
        if type(checkpoint[key]) is not bool:
            raise ValueError(f'Checkpoint {key} must be true or false')
    if checkpoint['kind'] == 'resume':
        required = {'optimizer_state_dict', 'scaler_state_dict', 'scheduler_state_dict', 'settings',
                    'best_validation_loss', 'best_candidate', 'stop_reason', 'validation_loss', 'validation'}
        if required - checkpoint.keys():
            raise ValueError('Incomplete resumable checkpoint')
        for key in ('validation_loss', 'best_validation_loss'):
            value = checkpoint[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Checkpoint {key} must be finite and nonnegative')
        if checkpoint['stop_reason'] not in (None, 'learning_rate_minimum'):
            raise ValueError('Invalid checkpoint stop reason')
    if resume and checkpoint['kind'] != 'resume':
        raise ValueError('A model-only candidate cannot resume training')
    if checkpoint['git_commit'] != (revision() if expected_commit is None else expected_commit):
        raise ValueError('Checkpoint Git commit does not match the run')
    validate_tensor_health(checkpoint)
    return checkpoint


def load_checkpoint(path, device='cpu', *, resume=False, expected_commit=None):
    return validate_checkpoint(torch.load(path, map_location=device, weights_only=True),
                               resume=resume, expected_commit=expected_commit)
