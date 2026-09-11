"""Bounded CUDA input-pipeline benchmark; leaves production data untouched.

PYTHONPATH=src python scripts/benchmark_pipeline.py --output /tmp/dw93net-benchmark
Synthetic dense products match current default shapes, not stimulus statistics.
Temporary data lives on the output filesystem and is removed after the run.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stimuli.load_flow_xt_data import ReichardtDataset
from features.dog_motion import DoGMotionBank
from models.pure_translation import PureTranslationNet


class PackedDataset(Dataset):
    def __init__(self, path):
        self.path = path
        array = np.load(path, mmap_mode='r')
        self.length = len(array)
        self.array = None

    def __getstate__(self):
        return dict(path=self.path, length=self.length, array=None)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode='r')
        x = torch.from_numpy(self.array[i].copy())
        # Retain the same per-sample finite-value validation as the file loader.
        assert torch.isfinite(x).all()
        return x.float(), torch.zeros(3)


def disk_read_bytes(pid='self'):
    return int(dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())['read_bytes'])


def evict(paths):
    # Only advise eviction of these benchmark files, never global drop_caches.
    for path in paths:
        with open(path, 'rb') as f:
            os.fsync(f.fileno())
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=384)
    parser.add_argument('--batches', type=int, default=24)
    args = parser.parse_args()
    assert torch.version.cuda and torch.version.hip is None
    assert torch.cuda.is_available()
    assert '3080 Ti' in torch.cuda.get_device_name(0)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    default_threads = torch.get_num_threads()
    bank = DoGMotionBank()
    shape = (bank.n_output_channels, 1, 7200 - bank.metadata["crop"]["left"] - bank.metadata["crop"]["right"])
    report = dict(architecture=PureTranslationNet.architecture, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  shape=shape, batch_size=12, samples=args.samples,
                  default_cpu_threads=default_threads, synthetic=True,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_tf32=torch.backends.cudnn.allow_tf32, runs={})

    def publish(name, result):
        report['runs'][name] = result
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, json.dumps(result), flush=True)

    model = PureTranslationNet(bank.dx_values).cuda().train()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def step(x, y):
        optimizer.zero_grad()
        pred = model(x)
        loss = sum(torch.nn.functional.mse_loss(pred[:, i], y[:, i]) / scale**2
                   for i, scale in enumerate((.02, .02)))
        loss.backward()
        optimizer.step()
        return loss.item()  # Same end-of-step synchronization as training.

    gx = torch.rand((12, *shape), device='cuda')
    gy = torch.zeros((12, 3), device='cuda')
    for _ in range(12):
        step(gx, gy)

    def pipeline(name, dataset, batches, workers=0, cold_paths=None):
        if cold_paths:
            evict(cold_paths)
        loader = DataLoader(dataset, batch_size=12, shuffle=True,
                            num_workers=workers, pin_memory=workers > 0,
                            generator=torch.Generator().manual_seed(42))
        iterator = iter(loader)
        # Worker startup is deliberately excluded from steady-state timing.
        if workers:
            next(iterator)
        torch.cuda.synchronize()
        worker_pids = [w.pid for w in iterator._workers] if workers else []
        def total_io():
            return disk_read_bytes() + sum(disk_read_bytes(pid) for pid in worker_pids)
        before_io = total_io()
        rows = []
        for _ in range(batches):
            t0 = time.perf_counter()
            x, y = next(iterator)
            t1 = time.perf_counter()
            x, y = x.cuda(), y.cuda()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            step(x, y)
            t3 = time.perf_counter()
            rows.append([(t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000])
        arr = np.asarray(rows)
        publish(name, dict(cpu_threads=torch.get_num_threads(), workers=workers,
                          batches=batches, mean_load_ms=float(arr[:, 0].mean()),
                          mean_transfer_ms=float(arr[:, 1].mean()),
                          mean_train_ms=float(arr[:, 2].mean()),
                          mean_total_ms=float(arr.sum(1).mean()),
                          median_total_ms=float(np.median(arr.sum(1))),
                          p90_total_ms=float(np.percentile(arr.sum(1), 90)),
                          process_and_worker_disk_read_bytes=total_io()-before_io))

    with tempfile.TemporaryDirectory(prefix='samples-', dir=args.output) as temp:
        root = Path(temp)
        torch.set_num_threads(2)
        array = np.lib.format.open_memmap(root / 'packed.npy', mode='w+', dtype=np.float16,
                                         shape=(args.samples, *shape))
        folder = root / 'products' / 'train'
        folder.mkdir(parents=True)
        (folder.parent / 'reichardt.json').write_text(json.dumps(bank.metadata))
        for i in range(args.samples):
            x = (torch.randint(0, 256, shape).float() / 255).square().half()
            array[i] = x.numpy()
            torch.save(x, folder / f'ndx=0_ndz=0_nrot=0_id={i}.pt')
        array.flush()
        del array
        dataset = ReichardtDataset(folder)
        paths = list(folder.glob('*.pt'))
        publish('storage_torch.float16', dict(files=len(paths),
                bytes=sum(p.stat().st_size for p in paths),
                allocated=sum(p.stat().st_blocks*512 for p in paths)))
        publish('storage_packed_float16', dict(bytes=(root / 'packed.npy').stat().st_size))
        torch.set_num_threads(default_threads)
        pipeline('files_float16_default_threads', dataset, 8)
        torch.set_num_threads(2)
        for repeat in range(3):
            pipeline(f'files_float16_warm_{repeat}', dataset, args.batches)
            pipeline(f'packed_float16_warm_{repeat}', PackedDataset(root / 'packed.npy'), args.batches)
        pipeline('files_torch.float16_cold', dataset, args.batches, cold_paths=paths)
        pipeline('packed_float16_cold', PackedDataset(root / 'packed.npy'), args.batches,
                 cold_paths=[root / 'packed.npy'])
        pipeline('files_float16_two_workers', dataset, args.batches, workers=2)
        torch.set_num_threads(default_threads)
        for repeat in range(3):
            pipeline(f'files_float16_workers_default_threads_cold_{repeat}',
                     dataset, args.batches, workers=2,
                     cold_paths=paths)
            pipeline(f'packed_float16_workers_default_threads_cold_{repeat}',
                     PackedDataset(root / 'packed.npy'), args.batches, workers=2,
                     cold_paths=[root / 'packed.npy'])
            pipeline(f'files_float16_four_workers_default_threads_cold_{repeat}',
                     dataset, args.batches, workers=4,
                     cold_paths=paths)

        for threads in (default_threads, 2):
            torch.set_num_threads(threads)
            timings = []
            for _ in range(48):
                start = time.perf_counter()
                step(gx, gy)
                timings.append((time.perf_counter()-start)*1000)
            publish(f'gpu_resident_threads_{threads}', dict(mean_ms=statistics.mean(timings),
                                                         median_ms=statistics.median(timings)))
        torch.set_num_threads(2)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(12):
                step(gx, gy)
        # CPU operation entries attribute the same CUDA work to their parent;
        # sum CUDA entries only, otherwise device time is counted twice.
        operations = sorted((e for e in prof.key_averages()
                             if e.device_type == torch.autograd.DeviceType.CUDA),
                            key=lambda e: e.self_device_time_total, reverse=True)
        publish('gpu_profile', dict(
            self_device_ms_per_batch=sum(e.self_device_time_total for e in operations)/12000,
            top_operations=[dict(name=e.key, self_device_ms_per_batch=e.self_device_time_total/12000)
                            for e in operations[:8]]))
        for dtype in (torch.float32, torch.float16):
            for pinned in (False, True):
                host = torch.rand((12, *shape), dtype=dtype, pin_memory=pinned)
                times = []
                for i in range(60):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    transferred = host.cuda(non_blocking=pinned).float()
                    torch.cuda.synchronize()
                    if i >= 10:
                        times.append((time.perf_counter()-start)*1000)
                publish(f'transfer_{dtype}_pinned_{pinned}', dict(mean_ms=statistics.mean(times),
                        bytes=host.numel()*host.element_size(), includes_gpu_float32_cast=True))
    print('Finished:', args.output / 'results.json', flush=True)


if __name__ == '__main__':
    main()
