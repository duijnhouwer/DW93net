"""One checkpoint schema and explicit, conservative source compatibility rules.

Git is provenance, not compatibility. The lists below deliberately over-invalidate
on edits to relevant modules; keep them aligned when adding local dependencies.
No legacy schemas or missing-field defaults are accepted.
"""
import hashlib
from pathlib import Path

import torch

from stimuli.metadata import DATA_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'dw93net-checkpoint-v1'
MODEL_SCHEMA = 'pure-translation-v1'
INFERENCE_SOURCES = (
    'src/checkpoints.py', 'src/models/pure_translation.py',
    'src/features/reichardt.py', 'src/features/dog_motion.py',
    'src/stimuli/load_flow_xt_data.py', 'src/stimuli/metadata.py',
    'src/evaluation.py',
)
TRAINING_SOURCES = INFERENCE_SOURCES + (
    'src/train/train_flow_xt_net.py', 'src/stimuli/generate_flow_xt_data.py',
    'src/stimuli/dataset_identity.py', 'scripts/run_training_recipe.py',
    'scripts/parallel_stimuli.py',
)


def code_identity(names):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def compatibility():
    return {'model_schema': MODEL_SCHEMA, 'data_schema': DATA_SCHEMA,
            'inference_code': code_identity(INFERENCE_SOURCES),
            'training_code': code_identity(TRAINING_SOURCES)}


def validate_checkpoint(checkpoint, *, resume=False):
    common = {'schema', 'kind', 'compatibility', 'data_contract', 'git_commit',
              'global_epoch', 'training_set_index', 'training_seed', 'epoch_within_set',
              'epochs_per_set', 'training_loss', 'learning_rate', 'detector',
              'n_input_channels', 'model_state_dict'}
    if not isinstance(checkpoint, dict) or common - checkpoint.keys():
        raise ValueError('Incomplete checkpoint schema; legacy checkpoints are not supported')
    if checkpoint['schema'] != SCHEMA or checkpoint['kind'] not in ('candidate', 'resume'):
        raise ValueError('Unsupported checkpoint schema or kind')
    current = compatibility()
    if not isinstance(checkpoint['compatibility'], dict) or current.keys() - checkpoint['compatibility'].keys():
        raise ValueError('Incomplete checkpoint compatibility metadata')
    keys = ('model_schema', 'data_schema', 'inference_code') + (('training_code',) if resume else ())
    if checkpoint['kind'] == 'resume':
        required = {'optimizer_state_dict', 'scaler_state_dict', 'scheduler_state_dict', 'settings'}
        if required - checkpoint.keys():
            raise ValueError('Incomplete resumable checkpoint')
    if resume and checkpoint['kind'] != 'resume':
        raise ValueError('A model-only candidate cannot resume training')
    if any(checkpoint['compatibility'].get(key) != current[key] for key in keys):
        raise ValueError('Checkpoint is incompatible with the current model/data schema or code')
    return checkpoint


def load_checkpoint(path, device='cpu', *, resume=False):
    return validate_checkpoint(torch.load(path, map_location=device, weights_only=True), resume=resume)
