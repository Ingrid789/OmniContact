"""Run one GPU training command, record device/process samples and elapsed wall time."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def query(kind, fields, gpu):
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), f"--query-{kind}={fields}",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=8, check=True,
    )
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="Physical nvidia-smi GPU index/UUID")
    parser.add_argument("--interval", type=float, default=2.)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("provide a training command and a positive sampling interval")
    gpu_fields = "uuid,name,memory.total,memory.used,utilization.gpu"
    uuid, name, total, used_start, _ = query("gpu", gpu_fields, args.gpu)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Retain the previous session's samples and final timing after a resume.
    started = datetime.now(timezone.utc)
    stem = started.strftime("gpu-%Y%m%dT%H%M%S%fZ")
    start = time.monotonic()
    process = subprocess.Popen(command)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda received, frame: process.send_signal(received))
    summary = dict(start_utc=started.isoformat(), pid=process.pid, gpu=name, gpu_uuid=uuid,
                   gpu_total_mib=int(total), initial_device_used_mib=int(used_start),
                   command=command, samples=0, sample_errors=0, peak_process_mib=0,
                   peak_device_mib=0)
    json_path = args.output_dir / (stem + ".json")

    def save():
        tmp = json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(tmp, json_path)

    save()
    busy_sum = 0
    with (args.output_dir / (stem + ".csv")).open("x", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(("wall_seconds", "device_used_mib", "process_used_mib", "device_gpu_busy_pct"))
        while process.poll() is None:
            try:
                _, _, _, used, busy = query("gpu", gpu_fields, args.gpu)[0]
                apps = query("compute-apps", "pid,used_memory", args.gpu)
                own = sum(int(mem) for pid, mem in apps if pid == str(process.pid))
                writer.writerow((round(time.monotonic()-start, 3), used, own, busy))
                summary["samples"] += 1
                busy_sum += int(busy)
                summary.update(peak_process_mib=max(summary["peak_process_mib"], own),
                               peak_device_mib=max(summary["peak_device_mib"], int(used)))
                summary["mean_device_gpu_busy_pct"] = round(busy_sum / summary["samples"], 2)
                summary["peak_process_memory_pct"] = round(100 * summary["peak_process_mib"] / int(total), 2)
                summary["peak_device_memory_pct"] = round(100 * summary["peak_device_mib"] / int(total), 2)
            except (subprocess.SubprocessError, OSError, IndexError, ValueError):
                summary["sample_errors"] += 1
            summary["wall_seconds"] = round(time.monotonic()-start, 3)
            save()
            try:
                process.wait(timeout=args.interval)
            except subprocess.TimeoutExpired:
                pass
    summary.update(exit_code=process.returncode, end_utc=datetime.now(timezone.utc).isoformat(),
                   wall_seconds=round(time.monotonic()-start, 3))
    save()
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
