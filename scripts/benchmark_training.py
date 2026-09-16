"""Run the recipe with timestamped output and one-second Linux resource samples."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amd-device', type=Path,
                        help='AMD GPU sysfs PCI device directory for telemetry')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    start = time.monotonic()
    def stamp():
        return dict(elapsed_s=round(time.monotonic() - start, 3),
                    utc=datetime.now(timezone.utc).isoformat())
    metadata = dict(stamp(), cpu_count=os.cpu_count(),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                   text=True).strip(),
                    recipe=(root / 'recipe/translation_only.toml').read_text())
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src')
    process = subprocess.Popen([sys.executable, '-u', 'scripts/run_training_recipe.py'],
                               cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)

    def capture():
        with (args.output / 'training.log').open('w') as stream:
            for line in process.stdout:
                stream.write(f'{time.monotonic() - start:.3f}s {line}')
                stream.flush()
                print(line, end='', flush=True)

    thread = threading.Thread(target=capture)
    thread.start()
    try:
        with (args.output / 'resources.jsonl').open('w') as stream:
            while process.poll() is None:
                sample = stamp()
                sample['cpu_ticks'] = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
                sample['load_average'] = os.getloadavg()
                sample['memory_kib'] = {line.split(':')[0]: int(line.split()[1])
                                        for line in Path('/proc/meminfo').read_text().splitlines()}
                sample['diskstats'] = Path('/proc/diskstats').read_text()
                try:
                    if args.amd_device:
                        device = args.amd_device
                        hwmon = next((device / 'hwmon').glob('hwmon*'))
                        def number(path):
                            return float(path.read_text().strip())
                        sample['gpu_rows'] = [[0, 'Radeon AI PRO R9700',
                            number(device / 'gpu_busy_percent'), None,
                            number(device / 'mem_info_vram_used') / 1048576,
                            number(device / 'mem_info_vram_total') / 1048576,
                            number(hwmon / 'power1_average') / 1000000,
                            number(hwmon / 'temp1_input') / 1000]]
                        sample['amd_temperatures_c'] = {
                            p.with_name(p.name.replace('_input', '_label')).read_text().strip(): number(p)/1000
                            for p in hwmon.glob('temp*_input')}
                        result = None
                    else:
                        result = subprocess.run(['nvidia-smi',
                        '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                    if result is not None:
                        sample['gpu_rows'] = list(csv.reader(result.stdout.splitlines()))
                        if result.returncode:
                            sample['gpu_error'] = result.stderr
                except (OSError, subprocess.TimeoutExpired) as error:
                    sample['gpu_error'] = str(error)
                stream.write(json.dumps(sample) + '\n')
                stream.flush()
                time.sleep(1)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        thread.join()
        (args.output / 'completion.json').write_text(json.dumps(
            dict(stamp(), returncode=process.returncode), indent=2))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
