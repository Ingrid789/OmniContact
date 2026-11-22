"""Compare real MuJoCo validation episodes at a shared V3 iteration."""

import argparse
import json
from pathlib import Path
import sys


def read_iteration(run, iteration):
    path = run / "metrics.jsonl"
    if not path.is_file():
        raise ValueError(f"{path} 尚无训练日志")
    match = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith("\n"):  # A running trainer may be appending this line.
                    break
                raise ValueError(f"{path} 第 {line_number} 行无法解析") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行缺少指标记录")
            if row.get("iteration") == iteration:
                match = row  # A resumed session can log the same iteration again.
    if match is None:
        raise ValueError(f"{path} 尚无第 {iteration} 轮记录")
    return match


def outcomes(row, source, run):
    keys = [f"eval/{source}/{key}" for key in ("done", "success", "drop")]
    if any(type(row.get(k)) is not int for k in keys):
        raise ValueError(f"{run} 缺少 {source} 的完整验证回合计数")
    done, success, drop = (row[k] for k in keys)
    if done <= 0 or not (0 <= success <= done and 0 <= drop <= done):
        raise ValueError(f"{run} 的 {source} 验证计数无效或无已结束回合")
    return done, success, drop


def report(with_noise, without_noise, iteration):
    runs = (with_noise, without_noise)
    records = [read_iteration(run, iteration) for run in runs]
    sources = [("npz", "NPZ 示范"), ("cfgen", "CF-Gen 参考")]
    recovery_counts = [row.get("eval/recovery/done") for row in records]
    if all(type(n) is int and n > 0 for n in recovery_counts):
        sources.append(("recovery", "掉箱后恢复"))

    lines = [
        f"## V3 第 {iteration} 轮 MuJoCo 验证",
        "",
        "| 参考来源 | CF-Gen 目标扰动 | 已结束回合 | 成功数（成功率） | 掉箱数（掉箱率） |",
        "|---|---|---:|---:|---:|",
    ]
    small_sample = False
    for source, title in sources:
        for run, record, label in zip(runs, records, ("有噪声", "无噪声")):
            done, success, drop = outcomes(record, source, run)
            lines.append(f"| {title} | {label} | {done} | {success}（{success / done:.1%}） "
                         f"| {drop}（{drop / done:.1%}） |")
            small_sample |= done < 10
    if small_sample:
        lines.extend(("", "部分分组少于 10 个已结束回合，单个回合就会明显改变百分比。"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="比较两组 V3 的同轮真实 MuJoCo 验证结果")
    parser.add_argument("with_noise", type=Path, help="有 CF-Gen 落点扰动的实验目录")
    parser.add_argument("without_noise", type=Path, help="无 CF-Gen 落点扰动的实验目录")
    parser.add_argument("--iteration", type=int, default=6000, help="共同验证轮数，默认 6000")
    args = parser.parse_args()
    if args.iteration < 1:
        parser.error("--iteration 须为正整数")
    try:
        print(report(args.with_noise, args.without_noise, args.iteration))
    except (ValueError, OSError) as exc:
        print(f"无法比较：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
