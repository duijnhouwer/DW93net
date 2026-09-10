"""Generate ten reproducible translation stimuli, four-polarity caches and PNG previews."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch

from data.generate_flow_xt_data import StimConfig, generate_flow_xt_dataset
from data.precompute_reichardt import precompute_dataset


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = StimConfig()
    settings = dict(n_samples=10, seed=42, speed_range=[.001, .015],
                    heading_deg_range=[-180, 180],
                    rot_deg_range=[0, 0], split=[.8, .1, .1], config=asdict(config))
    (root / 'generation.json').write_text(json.dumps(settings, indent=2) + '\n')
    source = Path(generate_flow_xt_dataset(output_root=str(root / 'raw'),
                                         **{**settings, 'config': config}))
    cache = root / 'products' / source.name
    metadata = precompute_dataset(source, cache)
    previews = root / 'previews'
    previews.mkdir()
    records = []
    for raw_path in sorted(source.rglob('*.pt')):
        relative = raw_path.relative_to(source)
        product_path = cache / relative
        raw = torch.load(raw_path, weights_only=True)
        products = torch.load(product_path, weights_only=True)
        assert raw.shape == (1, 2, 7200) and raw.dtype == torch.uint8
        assert products.shape == (484, 1, 7062) and products.dtype == torch.float16
        assert torch.isfinite(products).all() and (products >= 0).all()
        # Display only: max across six neighboring X pixels keeps sparse responses
        # visible. Stored tensors retain every original position and value.
        raw_view = raw[0].numpy().reshape(2, 1200, 6).max(axis=-1)
        maps = products[:, 0].float().numpy().reshape(4, 121, 1177, 6).max(axis=-1)
        fig, axes = plt.subplots(5, 1, figsize=(15, 10), height_ratios=[.8, 2, 2, 2, 2],
                                 constrained_layout=True)
        axes[0].imshow(raw_view, cmap='gray', vmin=0, vmax=255, aspect='auto',
                       extent=[0, 7200, 1.5, -.5], interpolation='nearest')
        axes[0].set_yticks([0, 1], ['Frame 0', 'Frame 1'])
        axes[0].set_title(raw_path.stem, fontsize=9)
        norm = PowerNorm(gamma=.35, vmin=0, vmax=max(float(maps.max()), 1e-8))
        for ax, values, name in zip(axes[1:], maps, metadata['polarity_order']):
            shown = ax.imshow(values, origin='lower', aspect='auto', cmap='magma', norm=norm,
                              extent=[68.5, 7130.5, -60.5, 60.5], interpolation='nearest')
            ax.set_ylabel(name + '\ndX (px/step)')
            ax.set_yticks([-60, -30, 0, 30, 60])
        axes[-1].set_xlabel('Present-frame X coordinate (pixels)')
        fig.colorbar(shown, ax=list(axes[1:]), label='Product magnitude (power display, gamma=0.35)', shrink=.8)
        fig.suptitle('Four polarity pairs — display uses 6-pixel maxima; saved maps are full resolution', fontsize=12)
        preview = previews / (raw_path.stem + '.png')
        fig.savefig(preview, dpi=110)
        plt.close(fig)
        records.append(dict(raw=str(raw_path.relative_to(root)), products=str(product_path.relative_to(root)),
                            preview=str(preview.relative_to(root)), raw_bytes=raw_path.stat().st_size,
                            product_bytes=product_path.stat().st_size))
    assert len(records) == 10
    (root / 'files.json').write_text(json.dumps(records, indent=2) + '\n')
    (root / 'README.md').write_text(
        '# Ten translation stimuli and their product maps\n\n'
        'Seed 42; speed sampled from [0.001,0.015) scene units per step and heading '
        'from [-180,180) degrees; yaw=0. Default 120-degree, '
        '7200-pixel stimuli, 1000 world dots, X supersampling 255. Splits: 8 train, '
        '1 validate, 1 test. See `generation.json` for exact settings.\n\n'
        'Each raw `.pt`: uint8 [1,2,7200]. Each product `.pt`: float16 [484,1,7062]. '
        'Channel blocks: ON_ON 0–120, OFF_OFF 121–241, ON_OFF 242–362, '
        'OFF_ON 363–483. Within each block: dX=-60…+60.\n\n'
        'Products use luminance DoG sigma 1/3 pixels and all four polarity pairs. '
        'The product root contains `reichardt.json`, `inputs.json`, and `provenance/` '
        'with commit, diff and source recovery archive. `files.json` links all files.\n\n'
        'PNG previews show all four blocks. Display only: six-pixel maximum pooling '
        'and a gamma=0.35 intensity scale expose sparse weak responses; a common '
        'color scale is used within each preview. Saved tensors are unaltered.\n')
    print(json.dumps(dict(root=str(root), metadata=str(cache / 'reichardt.json'),
                         samples=len(records), product_bytes=sum(r['product_bytes'] for r in records),
                         first_preview=str(root / records[0]['preview'])), indent=2))


if __name__ == '__main__':
    main()
