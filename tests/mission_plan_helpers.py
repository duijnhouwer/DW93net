"""Shared mission plan fixtures; independent of test discovery and test modules."""
from scripts.mission_control.backend import run_mission_plan as module

TEST_RUN_NAME = 'smoke-20000101000000-' + 'c' * 40

DETECTOR = {'displacement_min_px': -60, 'displacement_max_px': 60,
            'displacement_step_px': 1, 'dog_sigma_center_px': 1.0,
            'dog_sigma_surround_px': 3.0}


def mission_plan_text(root):
    return f'''
version = 1
require_cuda = false
[network]
output_path = "{root}/output"
[stimuli]
speed_range = [0.001, 0.006]
heading_deg_range = [-180, 180]
rot_deg_range = [0, 0]
dog_bundle_size = 64
duffy_wurtz_superimposition_enable = false
duffy_wurtz_rotation_pct = 50
[detector]
displacement_min_px = -60
displacement_max_px = 60
displacement_step_px = 1
dog_sigma_center_px = 1.0
dog_sigma_surround_px = 3.0
[training_sets]
path = "{root}/stimuli/training"
number_of_sets = 3
samples_per_set = 5000
epochs_per_set = 10
start_seed = 4000
seed_increment = 1
shuffle_seed = 7000
[validation_set]
path = "{root}/stimuli/validation"
seed = 2000
n_samples = 1000
selection_metric = "validation_loss"
[duffy_wurtz_1993_set]
path = "{root}/stimuli/duffy_wurtz_1993_set"
seed = 1000
n_samples = 1000
duffy_wurtz_superimposition_enable = true
[test_set]
path = "{root}/stimuli/test"
seed = 3000
n_samples = 1000
[optimization]
batch_size = 64
allow_amp = true
allow_tp32 = true
profile_batches = 20
num_workers = 0
mirror_train_stimuli = true
[learning_rate]
initial = 0.001
scheduler = "reduce_on_plateau"
monitor = "validation_loss"
factor = 0.5
patience_epochs = 10
minimum = 0.000001
relative_improvement_threshold = 0.0001
reset_on_new_training_set = false
[checkpoints]
candidate_interval_epochs = 1
keep_previous_resumable = true
[evaluation]
batch_size = 64
num_workers = 0
shuffle = false
'''


def candidate_checkpoint(commit, epoch=1):
    from checkpoints import SCHEMA
    return dict(schema=SCHEMA, kind='candidate', allow_amp=True, allow_tp32=True,
                data_contract={}, git_commit=commit, global_epoch=epoch,
                training_set_index=0, training_seed=1, epoch_within_set=epoch,
                epochs_per_set=epoch, training_loss=.1, learning_rate=.001,
                validation={}, validation_loss=1., best_validation_loss=1.,
                best_candidate='epoch-000001.pt', stop_reason=None,
                detector={}, n_input_channels=3, model_state_dict={})
