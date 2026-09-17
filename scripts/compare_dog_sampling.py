"""Paired CPU experiment: DoG before/after box downsampling (requires SciPy)."""
import argparse
import json
import time
import tomllib
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import fft
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from features.dog_motion import DoGFilter
from stimuli.generate_flow_xt_data import StimConfig, sample_motion, generate_flow_xt_stim


class FFTDoG:
    """Linear, valid convolution with a cached real FFT of the signed DoG.

    Gaussian normalization is float64, then weights/arithmetic are float32,
    matching DoGFilter. Both Gaussians use the surround support. The symmetric
    kernel makes convolution and PyTorch correlation identical mathematically.
    """
    def __init__(self, width, center, surround, workers):
        self.radius = int(np.ceil(3 * surround))
        self.width, self.workers = width, workers
        self.length = fft.next_fast_len(width + 2 * self.radius, real=True)
        x = np.arange(-self.radius, self.radius + 1, dtype=np.float64)
        weights = []
        for sigma in (center, surround):
            w = np.exp(-.5 * (x / sigma) ** 2)
            weights.append((w / w.sum()).astype(np.float32))
        self.kernel = weights[0] - weights[1]
        self.spectrum = fft.rfft(self.kernel, n=self.length, workers=workers)

    def __call__(self, frames):
        spectrum = fft.rfft(frames, n=self.length, axis=-1, workers=self.workers)
        spectrum *= self.spectrum
        full = fft.irfft(spectrum, n=self.length, axis=-1, workers=self.workers,
                         overwrite_x=True)
        return full[:, 2 * self.radius:self.width]


def self_check():
    """Check FFT alignment and accuracy independently against direct convolution."""
    rng = np.random.default_rng(10)
    errors = []
    for factor in (1, 7):
        frames = rng.random((2, 97 * factor), dtype=np.float32)
        direct = DoGFilter(sigma_center=factor, sigma_surround=3 * factor)(
            torch.from_numpy(frames))[0, 0].numpy()
        transformed = FFTDoG(frames.shape[-1], factor, 3 * factor, 1)(frames)
        np.testing.assert_allclose(transformed, direct, atol=1e-6, rtol=1e-5)
        errors.append(float(np.max(np.abs(transformed - direct))))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--supersampling', type=int, default=1023)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.supersampling, args.threads) < 1:
        parser.error('samples, supersampling, and threads must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    verification = self_check()
    recipe = tomllib.loads((ROOT / 'recipe/translation_only.toml').read_text())
    stim, detector = recipe['stimuli'], recipe['detector']
    cfg = StimConfig(**{k: v for k, v in stim.items() if k not in
        ('speed_range', 'heading_deg_range', 'rot_deg_range', 'dog_bundle_size')})
    cfg.x_supersampling = args.supersampling
    if cfg.superimpose:
        raise ValueError('This experiment requires non-superimposed stimuli')
    s, width = args.supersampling, cfg.stim_wid_px
    center, surround = detector['dog_sigma_center_px'], detector['dog_sigma_surround_px']
    low_filter = DoGFilter(sigma_center=center, sigma_surround=surround)
    radius = low_filter.radius
    high_filter = FFTDoG(width * s, center * s, surround * s, args.threads)
    if high_filter.radius != radius * s:
        raise ValueError('Kernel support must align exactly with output pixel blocks')
    low_fft = FFTDoG(width, center, surround, args.threads)
    shape = (args.samples, 2, width - 2 * radius)
    post = np.lib.format.open_memmap(out / 'dog_post.npy', mode='w+', dtype=np.float32, shape=shape)
    pre = np.lib.format.open_memmap(out / 'dog_pre.npy', mode='w+', dtype=np.float32, shape=shape)
    inputs = np.lib.format.open_memmap(out / 'xt_downsampled.npy', mode='w+', dtype=np.float32,
                                     shape=(args.samples, 2, width))
    labels = np.zeros((args.samples, 3), dtype=np.float32)
    # Warm both filtering implementations; exclude setup/warmup from timers.
    low_filter(torch.zeros(2, width))
    high_filter(np.zeros((2, width * s), dtype=np.float32))
    rng_low, rng_high = np.random.default_rng(args.seed), np.random.default_rng(args.seed)
    timing = {k: [] for k in ('render_low', 'render_high', 'dog_post', 'dog_pre', 'downsample_pre')}
    rows = []
    fft_noise_max = 0.
    started = time.perf_counter()
    print(f'{args.samples} pairs; width={width}, high width={width*s}; CPU threads={args.threads}', flush=True)
    for i in range(args.samples):
        motions = [sample_motion(rng, stim['speed_range'], stim['heading_deg_range'],
                                 stim['rot_deg_range']) for rng in (rng_low, rng_high)]
        assert motions[0] == motions[1]
        dx, dz, rot = motions[0]
        t = time.perf_counter()
        low = generate_flow_xt_stim([dx, dz], rot, cfg, rng_low).numpy()
        timing['render_low'].append(time.perf_counter() - t)
        t = time.perf_counter()
        high = generate_flow_xt_stim([dx, dz], rot, cfg, rng_high, downsample=False).numpy()
        timing['render_high'].append(time.perf_counter() - t)
        # Exact equality verifies seed, labels, placement, and box alignment.
        np.testing.assert_array_equal(high.reshape(2, width, s).mean(axis=2, dtype=np.float32), low)
        t = time.perf_counter()
        a = low_filter(torch.from_numpy(low))[0, 0].numpy()
        timing['dog_post'].append(time.perf_counter() - t)
        t = time.perf_counter()
        high_dog = high_filter(high)
        timing['dog_pre'].append(time.perf_counter() - t)
        t = time.perf_counter()
        b = high_dog.reshape(2, width - 2 * radius, s).mean(axis=2, dtype=np.float32)
        timing['downsample_pre'].append(time.perf_counter() - t)
        if i < 10:
            fft_noise_max = max(fft_noise_max, float(np.max(np.abs(low_fft(low) - a))))
        difference = b.astype(np.float64) - a
        reference = a.astype(np.float64)
        rms = float(np.sqrt(np.mean(difference ** 2)))
        reference_rms = float(np.sqrt(np.mean(reference ** 2)))
        rows.append(dict(sample=i, rmse=rms, reference_rms=reference_rms,
                         relative_rmse=rms / reference_rms, mae=float(np.mean(np.abs(difference))),
                         max_abs=float(np.max(np.abs(difference)))))
        post[i], pre[i], inputs[i], labels[i] = a, b, low, motions[0]
        del high_dog, high
        if (i + 1) % 25 == 0 or i + 1 == args.samples:
            elapsed = time.perf_counter() - started
            print(f'{i+1}/{args.samples}; elapsed {elapsed:.1f}s; ETA '
                  f'{elapsed/(i+1)*(args.samples-i-1):.1f}s', flush=True)
    for array in (pre, post, inputs):
        array.flush()
    np.save(out / 'labels.npy', labels)
    # Production uses batches of up to 16: also measure that low-res baseline.
    batched_seconds = 0.
    low_filter(torch.from_numpy(np.array(inputs[:min(16, args.samples)]))[:, None])
    for i in range(0, args.samples, 16):
        batch = torch.from_numpy(np.array(inputs[i:i+16]))[:, None]
        t = time.perf_counter()
        low_filter(batch)
        batched_seconds += time.perf_counter() - t
    stats = dict(
        samples=args.samples, seed=args.seed, config=vars(cfg), center_sigma=center,
        surround_sigma=surround, output_shape=list(shape), device='CPU', threads=args.threads,
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__),
        fft_direct_self_check_max_abs=verification, low_resolution_fft_noise_max_abs=fft_noise_max,
        pooled_rmse=float(np.sqrt(np.mean([r['rmse']**2 for r in rows]))),
        pooled_reference_rms=float(np.sqrt(np.mean([r['reference_rms']**2 for r in rows]))),
        mean_abs_difference=float(np.mean([r['mae'] for r in rows])),
        max_abs_difference=max(r['max_abs'] for r in rows),
        pairs_with_difference=sum(r['max_abs'] > 0 for r in rows),
        timing={k: dict(total_s=float(np.sum(v)), median_ms=float(np.median(v)*1000),
                        mean_ms=float(np.mean(v)*1000)) for k, v in timing.items()},
        post_batched_16_total_s=batched_seconds,
        elapsed_s=time.perf_counter()-started,
        method='Post: existing float32 PyTorch DoG. Pre: float32 FFT convolution of signed DoG, then box mean. '
               'Gaussian weights normalized in float64. Same angular sigmas, valid crop, independent equal-seed streams. '
               'Filter timers exclude rendering, kernel setup, validation and disk I/O; warmup excluded.')
    stats['relative_rmse'] = stats['pooled_rmse']/stats['pooled_reference_rms']
    stats['filter_time_ratio'] = stats['timing']['dog_pre']['total_s']/stats['timing']['dog_post']['total_s']
    stats['filter_time_ratio_vs_batched_post'] = stats['timing']['dog_pre']['total_s']/batched_seconds
    (out / 'summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    (out / 'pairs.json').write_text(json.dumps(rows, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    worst = max(rows, key=lambda r: r['rmse'])['sample']
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    ax[0].hist([r['relative_rmse']*100 for r in rows], bins=min(40, len(rows)))
    ax[0].set(xlabel='Pair RMSE / post-DoG RMS (%)', ylabel='Number of pairs', title='Effect of filtering before downsampling')
    peak = int(np.argmax(np.abs(pre[worst, 0] - post[worst, 0])))
    left, right = max(0, peak-50), min(shape[-1], peak+51)
    ax[1].plot(np.arange(left,right), post[worst,0,left:right], label='DoG after downsampling')
    ax[1].plot(np.arange(left,right), pre[worst,0,left:right], '--', label='DoG before downsampling')
    ax[1].set(xlabel='Output pixel', ylabel='Signed contrast', title=f'Worst-RMSE pair {worst}, first frame: local detail')
    ax[1].legend()
    fig.savefig(out / 'comparison.png', dpi=160)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
