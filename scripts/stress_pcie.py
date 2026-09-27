"""Saturate the CUDA host-to-device path for a bounded PCIe link-state check."""

import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60,
                        help='How long to transfer data (default: 60 seconds)')
    parser.add_argument('--chunk-mib', type=int, default=256,
                        help='Pinned host buffer size per transfer (default: 256 MiB)')
    parser.add_argument('--copies-per-sync', type=int, default=8,
                        help='DMA copies queued before synchronization (default: 8)')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    if args.chunk_mib <= 0:
        parser.error('--chunk-mib must be positive')
    if args.copies_per_sync <= 0:
        parser.error('--copies-per-sync must be positive')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')

    device = torch.device('cuda')
    count = args.chunk_mib * 1024 * 1024 // 4
    host = torch.empty(count, dtype=torch.float32, pin_memory=True)
    gpu = torch.empty_like(host, device=device)
    stream = torch.cuda.Stream(device=device)

    # Touch the pages and initialize CUDA before starting the measurement.
    host.uniform_()
    with torch.cuda.stream(stream):
        gpu.copy_(host, non_blocking=True)
    stream.synchronize()

    print(f'Device: {torch.cuda.get_device_name(device)}')
    print(f'Transferring {args.chunk_mib} MiB chunks host -> GPU for {args.seconds:g}s')
    print(f'Queue depth: {args.copies_per_sync} copies per synchronization')
    print('This uses the CUDA DMA copy engine; it launches no compute kernels.')
    print('Press Ctrl+C to stop early.', flush=True)

    started = time.perf_counter()
    interval_started = started
    interval_bytes = total_bytes = 0
    try:
        while time.perf_counter() - started < args.seconds:
            with torch.cuda.stream(stream):
                for _ in range(args.copies_per_sync):
                    gpu.copy_(host, non_blocking=True)
            stream.synchronize()
            transferred = host.nbytes * args.copies_per_sync
            total_bytes += transferred
            interval_bytes += transferred
            now = time.perf_counter()
            if now - interval_started >= 1:
                gib_s = interval_bytes / (now - interval_started) / 2**30
                print(f'{now - started:6.1f}s  {gib_s:6.2f} GiB/s', flush=True)
                interval_started = now
                interval_bytes = 0
    except KeyboardInterrupt:
        print('\nStopped early.', flush=True)
    finally:
        stream.synchronize()

    elapsed = time.perf_counter() - started
    print(f'Average: {total_bytes / elapsed / 2**30:.2f} GiB/s over {elapsed:.2f}s')


if __name__ == '__main__':
    main()
