"""Measure one production-width GPU training batch without writing training artifacts."""
import argparse
import json
import time
from pathlib import Path

import torch
from run_training_recipe import load_recipe
from stimuli.generate_flow_xt_data import StimConfig
from features.reichardt import ReichardtBank
from models.flatland_flow import FlatlandFlowNet
from precision import configure_tf32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=Path(__file__).resolve().parents[1] /
                        'recipe/trans_plus_rot_panoramic.toml')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--float32', action='store_true', help='Measure the numerical fallback mode')
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    config = StimConfig(fov_deg=recipe['stimuli']['fov_deg'])
    if config.boundary != 'circular':
        raise ValueError('This preflight requires the panoramic recipe')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run on the training GPU')
    torch.set_num_threads(4)
    batch = args.batch_size or recipe['optimization']['batch_size']
    if batch < 1:
        raise ValueError('batch-size must be positive')
    amp = recipe['optimization']['allow_amp'] and not args.float32
    configure_tf32(recipe['optimization']['allow_tp32'] and not args.float32)
    detector = recipe['detector']
    spans = tuple(range(detector['displacement_min_px'], detector['displacement_max_px'] + 1,
                        detector['displacement_step_px']))
    bank = ReichardtBank(spans, boundary=config.boundary)
    free, total = torch.cuda.mem_get_info()
    report = dict(device=torch.cuda.get_device_name(), width=config.stim_wid_px,
                  channels=len(spans), batch_size=batch, amp=amp,
                  free_before_mib=free / 2**20, total_mib=total / 2**20)
    started = time.monotonic()
    try:
        model = FlatlandFlowNet(spans, boundary=config.boundary).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe['learning_rate']['initial'])
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        # Signed cached contrasts; renderer and DoG run on CPU during dataset generation.
        torch.manual_seed(42)
        contrast = torch.randn(batch, 1, 2, config.stim_wid_px, device='cuda') * .01
        torch.cuda.reset_peak_memory_stats()
        responses = bank(contrast, validate=False)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            prediction = model(responses)
        loss = prediction.float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite preflight loss or parameters')
        report.update(status='ok', loss=loss.item())
    except torch.OutOfMemoryError as error:
        report.update(status='out_of_memory', error=str(error))
    report.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  elapsed_s=time.monotonic() - started)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
