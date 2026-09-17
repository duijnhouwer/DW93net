"""Checkpoint structure and exact Git-commit matching."""

import torch

from run_revision import revision

SCHEMA = 'dw93net-checkpoint-v2'


def validate_checkpoint(checkpoint, *, resume=False, expected_commit=None):
    common = {'schema', 'kind', 'data_contract', 'git_commit',
              'global_epoch', 'training_set_index', 'training_seed', 'epoch_within_set',
              'epochs_per_set', 'training_loss', 'learning_rate', 'detector',
              'n_input_channels', 'model_state_dict'}
    if not isinstance(checkpoint, dict) or common - checkpoint.keys():
        raise ValueError('Incomplete checkpoint schema; legacy checkpoints are not supported')
    if checkpoint['schema'] != SCHEMA or checkpoint['kind'] not in ('candidate', 'resume'):
        raise ValueError('Unsupported checkpoint schema or kind')
    if checkpoint['kind'] == 'resume':
        required = {'optimizer_state_dict', 'scaler_state_dict', 'scheduler_state_dict', 'settings'}
        if required - checkpoint.keys():
            raise ValueError('Incomplete resumable checkpoint')
    if resume and checkpoint['kind'] != 'resume':
        raise ValueError('A model-only candidate cannot resume training')
    if checkpoint['git_commit'] != (revision() if expected_commit is None else expected_commit):
        raise ValueError('Checkpoint Git commit does not match the run')
    return checkpoint


def load_checkpoint(path, device='cpu', *, resume=False, expected_commit=None):
    return validate_checkpoint(torch.load(path, map_location=device, weights_only=True),
                               resume=resume, expected_commit=expected_commit)
