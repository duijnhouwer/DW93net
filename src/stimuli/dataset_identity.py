"""Exact run commit/settings and structural validation for managed DoG caches."""

import json
from pathlib import Path

import numpy as np
from stimuli.metadata import require_current_schema
from run_revision import revision
from features.spatial import resolve_boundary


def dataset_spec(role, section, stimuli):
    return {'version': 6, 'boundary': resolve_boundary(stimuli.get('fov_deg', 120)),
            'git_commit': section.get('_git_commit') or revision(),
            'role': role, 'seed': section['seed'], 'n_samples': section['n_samples'],
            'stimuli': stimuli, 'dog': {key: section['_detector'][key] for key in
                ('dog_sigma_center_px', 'dog_sigma_surround_px')}}


def storage_manifest(cache, role, n_samples):
    """Validate every index row and bundle header, without scanning tensor payloads.

    This detects missing/truncated bundles and inconsistent layouts, not same-size
    payload corruption. Header-only checks keep large-dataset restarts inexpensive.
    """
    cache = Path(cache)
    metadata_bytes = (cache / 'dog.json').read_bytes()
    index_bytes = (cache / role / 'bundles.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    require_current_schema(metadata)
    index = json.loads(index_bytes)
    shape = metadata.get('contrast_shape')
    if (metadata.get('storage_dtype') != 'torch.float32'
            or metadata.get('finite_validated') is not True
            or metadata.get('n_samples') != n_samples
            or not isinstance(shape, list) or len(shape) != 2 or shape[0] != 2
            or any(type(size) is not int or size <= 0 for size in shape)
            or not isinstance(index, list) or len(index) != n_samples):
        raise ValueError('Invalid dataset metadata or sample count')
    rows, sample_ids = {}, set()
    for entry in index:
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError('Invalid bundle index entry')
        sample, name, row = entry
        if (type(sample) is not int or sample < 0 or sample in sample_ids
                or not isinstance(name, str) or Path(name).name != name
                or not name.endswith('.npy') or type(row) is not int or row < 0):
            raise ValueError('Invalid or duplicate bundle index entry')
        sample_ids.add(sample)
        bundle_rows = rows.setdefault(name, set())
        if row in bundle_rows:
            raise ValueError('Duplicate bundle row')
        bundle_rows.add(row)
    if sample_ids != set(range(n_samples)):
        raise ValueError('Sample IDs must cover 0..n_samples-1')
    folder = cache / role
    if {p.name for p in folder.glob('*.npy')} != set(rows):
        raise ValueError('Bundle files do not match the index')
    dtype = np.dtype([('dog_contrast', np.float32, tuple(shape)),
                      ('target', np.float32, (3,))])
    bundles = {}
    for name, indexed_rows in sorted(rows.items()):
        path = folder / name
        if path.is_symlink():
            raise ValueError('Bundle must not be a symlink')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            size = path.stat().st_size
            if (array.ndim != 1 or array.dtype != dtype
                    or indexed_rows != set(range(len(array)))
                    or size != array.offset + array.nbytes):
                raise ValueError(f'Invalid bundle layout or count: {name}')
            bundles[name] = {'n_samples': len(array), 'size_bytes': size}
        finally:
            del array
    return {'metadata': metadata,
            'bundles': bundles}
