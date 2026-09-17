"""Generate ten reproducible signed DoG samples and online-correlation previews."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import torch

from features.reichardt import ReichardtBank
from stimuli.generate_flow_xt_data import StimConfig, generate_dog_dataset
from stimuli.load_flow_xt_data import DoGDataset


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = StimConfig()
    settings = dict(n_samples=10, seed=42, speed_range=[.001, .015],
                    heading_deg_range=[-180, 180], rot_deg_range=[0, 0],
                    role='train', config=asdict(config))
    (root / 'generation.json').write_text(json.dumps(settings, indent=2) + '\n')
    cache = Path(generate_dog_dataset(
        output_root=str(root), dog_bundle_size=4,
        **{**settings, 'config': config}))
    previews = root / 'previews'
    previews.mkdir()
    records = []
    bank = ReichardtBank()
    dataset = DoGDataset(cache / 'train')
    for index, sample_name in enumerate(dataset.sample_ids):
        _, bundle, row = dataset.bundle_index[index]
        contrast, target = dataset[index]
        correlations = bank(contrast, validate=False)
        assert contrast.shape == (2, 7182) and contrast.dtype == torch.float32
        assert correlations.shape == (121, 1, 7062)
        assert torch.isfinite(contrast).all() and torch.isfinite(correlations).all()

        contrast_view = contrast.numpy().reshape(2, 1197, 6).mean(axis=-1)
        maps = correlations[:, 0].numpy().reshape(121, 1177, 6).mean(axis=-1)
        magnitude = max(float(np.abs(maps).max()), 1e-8)
        fig, axes = plt.subplots(2, 1, figsize=(15, 7),
                                 height_ratios=[1, 4], constrained_layout=True)
        axes[0].imshow(contrast_view, cmap='coolwarm', aspect='auto',
                       extent=[9, 7191, 1.5, -.5], interpolation='nearest')
        axes[0].set_yticks([0, 1], ['Frame 0', 'Frame 1'])
        axes[0].set_title(sample_name, fontsize=9)
        shown = axes[1].imshow(
            maps, origin='lower', aspect='auto', cmap='coolwarm',
            norm=SymLogNorm(linthresh=magnitude / 1000, vmin=-magnitude, vmax=magnitude),
            extent=[68.5, 7130.5, -60.5, 60.5], interpolation='nearest')
        axes[1].set_ylabel('dX (pixels/step)')
        axes[1].set_xlabel('Present-frame X coordinate (pixels)')
        fig.colorbar(shown, ax=axes[1], label='Signed correlation')
        preview = previews / (Path(sample_name).stem + '.png')
        fig.savefig(preview, dpi=110)
        plt.close(fig)
        bundle_path = cache / 'train' / bundle
        records.append(dict(sample=sample_name,
                            dog_bundle=str(bundle_path.relative_to(root)),
                            bundle_row=row, target=target.tolist(),
                            preview=str(preview.relative_to(root)),
                            dog_bytes=contrast.numel() * contrast.element_size(),
                            ephemeral_correlation_bytes=(
                                correlations.numel() * correlations.element_size())))
    assert len(records) == 10
    (root / 'files.json').write_text(json.dumps(records, indent=2) + '\n')
    (root / 'README.md').write_text(
        '# Ten signed DoG stimuli and their online correlations\n\n'
        'Seed 42; speed sampled from [0.001,0.015) scene units per step and heading '
        'from [-180,180) degrees; yaw=0. Default 120-degree, 7200-pixel stimuli, '
        '1000 world dots, X supersampling 255. Role: train.\n\n'
        'Only float32 signed DoG contrast `[2,7182]` and float32 targets are stored. '
        'The signed `[-60,+60]` correlation maps are calculated in memory for each '
        'preview and are not written to disk. `dog.json` and `provenance/` record '
        'the cache semantics and recoverable source state.\n')
    print(json.dumps(dict(root=str(root), metadata=str(cache / 'dog.json'),
                         samples=len(records), dog_bytes=sum(r['dog_bytes'] for r in records),
                         first_preview=str(root / records[0]['preview'])), indent=2))


if __name__ == '__main__':
    main()
