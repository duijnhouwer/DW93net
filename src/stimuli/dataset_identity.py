"""Cheap source identity and structural validation for managed DoG caches."""

import hashlib
import json
from pathlib import Path

import numpy as np


def generation_identity():
    # Keep this list aligned with the renderer's local implementation dependencies.
    # Training/model changes must not invalidate an otherwise identical dataset.
    root = Path(__file__).resolve().parents[1]
    names = ('stimuli/generate_flow_xt_data.py', 'features/dog_motion.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


def dataset_spec(role, section, stimuli):
    return {'version': 3, 'generation_identity': generation_identity(),
            'role': role, 'seed': section['seed'], 'n_samples': section['n_samples'],
            'stimuli': stimuli, 'detector': section['_detector']}


def storage_manifest(cache, role, n_samples):
    """Validate every index row and bundle header, without scanning tensor payloads.

    This detects missing/truncated bundles and inconsistent layouts, not same-size
    payload corruption. Header-only checks keep large-dataset restarts inexpensive.
    """
    cache = Path(cache)
    metadata_bytes = (cache / 'dog.json').read_bytes()
    index_bytes = (cache / role / 'bundles.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    index = json.loads(index_bytes)
    shape = metadata.get('contrast_shape')
    if (metadata.get('storage_format') != 'dog-contrast-bundles-v1'
            or metadata.get('storage_dtype') != 'torch.float32'
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
        if (not isinstance(sample, str) or sample in sample_ids
                or not isinstance(name, str) or Path(name).name != name
                or not name.endswith('.npy') or type(row) is not int or row < 0):
            raise ValueError('Invalid or duplicate bundle index entry')
        sample_ids.add(sample)
        bundle_rows = rows.setdefault(name, set())
        if row in bundle_rows:
            raise ValueError('Duplicate bundle row')
        bundle_rows.add(row)
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
    return {'dog_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
            'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
            'bundles': bundles}
