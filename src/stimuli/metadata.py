"""Scientific meaning comes from metadata, never directory or sample names."""
import json
import math
from pathlib import Path

import numpy as np

DATA_SCHEMA = 'dog-contrast-bundles-v2'


def bounds(spec):
    if spec['sampling'] == 'fixed':
        low = high = float(spec['value'])
    elif spec['sampling'] == 'uniform_half_open':
        low, high = float(spec['low_inclusive']), float(spec['high_exclusive'])
    else:
        raise ValueError('Unsupported motion sampling metadata')
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError('Invalid motion bounds in metadata')
    return low, high


def read_metadata(cache):
    metadata = json.loads((Path(cache) / 'dog.json').read_text())
    if metadata['storage_format'] != DATA_SCHEMA:
        raise ValueError('Unsupported dataset schema; regenerate this dataset')
    generation = metadata['stimulus_generation']
    if generation['format'] != 'polar-translation-stimuli-v2':
        raise ValueError('Unsupported stimulus metadata')
    speed = bounds(generation['speed_range'])
    heading = bounds(generation['heading_deg_range'])
    bounds(generation['rot_deg_range'])
    if speed[0] <= 0 or heading[0] < -180 or heading[1] > 180:
        raise ValueError('Invalid polar motion bounds')
    config = generation['stimulus_config']
    for key in ('stim_wid_px', 'fov_deg', 'near_exclusion_radius',
                'n_world_dots', 'superimpose', 'x_supersampling'):
        if key not in config:
            raise ValueError(f'Missing stimulus geometry: {key}')
    return metadata


def motion_ranges(generation):
    speed_low, speed_high = bounds(generation['speed_range'])
    heading_low, heading_high = bounds(generation['heading_deg_range'])
    rot_low, rot_high = bounds(generation['rot_deg_range'])

    def component_range(function, critical_angles):
        angles = [heading_low, heading_high]
        angles.extend(angle for angle in critical_angles if heading_low <= angle <= heading_high)
        values = [speed * function(np.deg2rad(angle))
                  for speed in (speed_low, speed_high) for angle in angles]
        return max(values) - min(values)

    return {'dx': component_range(np.sin, (-90, 90)),
            'dz': component_range(np.cos, (-180, 0, 180)), 'rot_deg': rot_high - rot_low}


def motion_scales(cache):
    generation = read_metadata(cache)['stimulus_generation']
    ranges = motion_ranges(generation)
    # A fixed speed and heading has zero span on both axes; use its physical
    # speed as the scale instead of inventing a value from a folder name.
    translation_scale = max(ranges['dx'], ranges['dz']) or bounds(generation['speed_range'])[1]
    return (ranges['dx'] or translation_scale, ranges['dz'] or translation_scale,
            ranges['rot_deg'] or 1.)


def data_contract(metadata):
    """Input meaning required by a model; ranges/seed/sample count may vary."""
    return {'schema': metadata['storage_format'],
            'geometry': metadata['stimulus_generation']['stimulus_config'],
            'git_commit': metadata['git_commit']}
