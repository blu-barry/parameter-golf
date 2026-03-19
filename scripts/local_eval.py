#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ENV = {
    "ITERATIONS": "200",
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_LOSS_EVERY": "0",
    "VAL_BATCH_SIZE": "524288",
    "VAL_PROGRESS_SECONDS": "5.0",
}

RUN_ID_RE = re.compile(r"^run_id:(.+)$")
TRAIN_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iterations>\d+) train_loss:(?P<train_loss>[0-9.]+) "
    r"train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms tok_s:(?P<tok_s>\d+)$"
)
VAL_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>[0-9.]+) "
    r"val_bpb:(?P<val_bpb>[0-9.]+) "
)
FINAL_EXACT_RE = re.compile(
    r"^final_int8_zlib_roundtrip_exact val_loss:(?P<final_val_loss>[0-9.]+) "
    r"val_bpb:(?P<final_val_bpb>[0-9.]+)$"
)
SERIALIZED_RE = re.compile(r"^serialized_model_int8_zlib:(?P<compressed_bytes>\d+) bytes ")
VAL_EVAL_DONE_RE = re.compile(r"^val_eval_done elapsed:(?P<seconds>[0-9.]+)s$")
FINAL_VAL_EVAL_DONE_RE = re.compile(r"^final_val_eval_done elapsed:(?P<seconds>[0-9.]+)s$")


@dataclass
class RunMetrics:
    label: str
    seed: int | None
    run_id: str
    log_path: str
    status: str
    exit_code: int | None = None
    train_loss: float | None = None
    train_step_avg_ms: float | None = None
    train_tok_s: int | None = None
    pre_quant_val_loss: float | None = None
    pre_quant_val_bpb: float | None = None
    final_val_loss: float | None = None
    final_val_bpb: float | None = None
    quant_gap: float | None = None
    compressed_bytes: int | None = None
    val_eval_seconds: float | None = None
    final_val_eval_seconds: float | None = None


def parse_label_seed(run_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<label>.+)_s(?P<seed>\d+)_\d{8}$", run_id)
    if match:
        return match.group("label"), int(match.group("seed"))
    return run_id, None


def parse_log(log_path: Path) -> RunMetrics:
    label, seed = parse_label_seed(log_path.stem)
    metrics = RunMetrics(label=label, seed=seed, run_id=log_path.stem, log_path=str(log_path), status="missing")
    if not log_path.exists():
        return metrics

    run_id = log_path.stem
    status = "partial"
    last_train: dict[str, str] | None = None
    last_val: dict[str, str] | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        run_match = RUN_ID_RE.match(line)
        if run_match:
            run_id = run_match.group(1)
            label, seed = parse_label_seed(run_id)
            metrics.label = label
            metrics.seed = seed
            metrics.run_id = run_id
            continue
        train_match = TRAIN_RE.match(line)
        if train_match:
            last_train = train_match.groupdict()
            continue
        val_match = VAL_RE.match(line)
        if val_match:
            last_val = val_match.groupdict()
            continue
        final_match = FINAL_EXACT_RE.match(line)
        if final_match:
            metrics.final_val_loss = float(final_match.group("final_val_loss"))
            metrics.final_val_bpb = float(final_match.group("final_val_bpb"))
            status = "complete"
            continue
        serialized_match = SERIALIZED_RE.match(line)
        if serialized_match:
            metrics.compressed_bytes = int(serialized_match.group("compressed_bytes"))
            continue
        val_done_match = VAL_EVAL_DONE_RE.match(line)
        if val_done_match:
            metrics.val_eval_seconds = float(val_done_match.group("seconds"))
            continue
        final_val_done_match = FINAL_VAL_EVAL_DONE_RE.match(line)
        if final_val_done_match:
            metrics.final_val_eval_seconds = float(final_val_done_match.group("seconds"))
            continue

    metrics.status = status
    if last_train is not None:
        metrics.train_loss = float(last_train["train_loss"])
        metrics.train_step_avg_ms = float(last_train["step_avg_ms"])
        metrics.train_tok_s = int(last_train["tok_s"])
    if last_val is not None:
        metrics.pre_quant_val_loss = float(last_val["val_loss"])
        metrics.pre_quant_val_bpb = float(last_val["val_bpb"])
    if metrics.pre_quant_val_bpb is not None and metrics.final_val_bpb is not None:
        metrics.quant_gap = metrics.final_val_bpb - metrics.pre_quant_val_bpb
    return metrics


def parse_env_overrides(pairs: list[str]) -> dict[str, str]:
    env = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing key in override: {pair}")
        env[key] = value.strip()
    return env


def now_suffix() -> str:
    return time.strftime("%H%M%S")


def print_run_summary(metrics: RunMetrics) -> None:
    print(
        f"{metrics.run_id}: status={metrics.status} "
        f"final_val_bpb={fmt(metrics.final_val_bpb)} "
        f"pre_quant_val_bpb={fmt(metrics.pre_quant_val_bpb)} "
        f"quant_gap={fmt(metrics.quant_gap)} "
        f"bytes={fmt(metrics.compressed_bytes)} "
        f"tok_s={fmt(metrics.train_tok_s)} "
        f"val_eval_s={fmt(metrics.val_eval_seconds)} "
        f"final_val_eval_s={fmt(metrics.final_val_eval_seconds)}"
    )


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}" if abs(value) < 1000 else f"{value:.1f}"


def completed_runs(metrics_list: Iterable[RunMetrics]) -> list[RunMetrics]:
    return [metrics for metrics in metrics_list if metrics.status == "complete" and metrics.final_val_bpb is not None]


def aggregate_metrics(metrics_list: list[RunMetrics]) -> dict[str, float] | None:
    complete = completed_runs(metrics_list)
    if not complete:
        return None

    finals = [metrics.final_val_bpb for metrics in complete if metrics.final_val_bpb is not None]
    gaps = [metrics.quant_gap for metrics in complete if metrics.quant_gap is not None]
    bytes_list = [metrics.compressed_bytes for metrics in complete if metrics.compressed_bytes is not None]
    tok_s = [metrics.train_tok_s for metrics in complete if metrics.train_tok_s is not None]
    out: dict[str, float] = {
        "runs": float(len(complete)),
        "mean_final_val_bpb": statistics.mean(finals),
        "min_final_val_bpb": min(finals),
        "max_final_val_bpb": max(finals),
    }
    if len(finals) > 1:
        out["range_final_val_bpb"] = max(finals) - min(finals)
    if gaps:
        out["mean_quant_gap"] = statistics.mean(gaps)
    if bytes_list:
        out["mean_compressed_bytes"] = statistics.mean(bytes_list)
    if tok_s:
        out["mean_tok_s"] = statistics.mean(tok_s)
    return out


def print_aggregate(metrics_list: list[RunMetrics]) -> None:
    aggregate = aggregate_metrics(metrics_list)
    if aggregate is None:
        print("No complete runs to aggregate.")
        return

    print("\nAggregate")
    print(f"runs={int(aggregate['runs'])}")
    print(f"mean_final_val_bpb={aggregate['mean_final_val_bpb']:.8f}")
    print(f"min_final_val_bpb={aggregate['min_final_val_bpb']:.8f}")
    print(f"max_final_val_bpb={aggregate['max_final_val_bpb']:.8f}")
    if "range_final_val_bpb" in aggregate:
        print(f"range_final_val_bpb={aggregate['range_final_val_bpb']:.8f}")
    if "mean_quant_gap" in aggregate:
        print(f"mean_quant_gap={aggregate['mean_quant_gap']:.8f}")
    if "mean_compressed_bytes" in aggregate:
        print(f"mean_compressed_bytes={aggregate['mean_compressed_bytes']:.1f}")
    if "mean_tok_s" in aggregate:
        print(f"mean_tok_s={aggregate['mean_tok_s']:.1f}")


def save_summary(metrics_list: list[RunMetrics], output_path: Path) -> None:
    output_path.write_text(json.dumps([asdict(metrics) for metrics in metrics_list], indent=2), encoding="utf-8")


def run_experiment(
    repo_root: Path,
    label: str,
    seeds: list[int],
    overrides: dict[str, str],
    *,
    dry_run: bool = False,
    keep_going: bool = False,
    write_summary: bool = True,
) -> tuple[list[RunMetrics], Path | None]:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    summary: list[RunMetrics] = []
    batch_suffix = now_suffix()

    for seed in seeds:
        run_id = f"{label}_s{seed}_{batch_suffix}"
        env = os.environ.copy()
        env.update(DEFAULT_ENV)
        env.update(overrides)
        env["RUN_ID"] = run_id
        env["SEED"] = str(seed)
        env["PYTHONUNBUFFERED"] = "1"
        log_path = logs_dir / f"{run_id}.txt"

        cmd = [sys.executable, "train_gpt_mlx.py"]
        print(f"\n=== Running {run_id} ===")
        print("Environment overrides:")
        merged = {**DEFAULT_ENV, **overrides, "SEED": str(seed), "RUN_ID": run_id}
        for key in sorted(merged):
            print(f"  {key}={merged[key]}")

        if dry_run:
            summary.append(parse_log(log_path))
            continue

        completed = subprocess.run(cmd, cwd=repo_root, env=env)
        metrics = parse_log(log_path)
        metrics.exit_code = completed.returncode
        if completed.returncode != 0 and metrics.status != "complete":
            metrics.status = "failed"
        summary.append(metrics)
        print_run_summary(metrics)

        if completed.returncode != 0 and not keep_going:
            print("Stopping after failure. Re-run with --keep-going to continue remaining seeds.")
            break

    summary_path = None
    if summary and write_summary:
        summary_path = logs_dir / f"{label}_summary_{batch_suffix}.json"
        save_summary(summary, summary_path)
        print(f"\nWrote summary: {summary_path}")
        for metrics in summary:
            print_run_summary(metrics)
        print_aggregate(summary)
    return summary, summary_path


def print_compare(baseline_metrics: list[RunMetrics], candidate_metrics: list[RunMetrics]) -> None:
    baseline_agg = aggregate_metrics(baseline_metrics)
    candidate_agg = aggregate_metrics(candidate_metrics)
    if baseline_agg is None or candidate_agg is None:
        print("Need at least one complete run in each set.")
        return

    print("Baseline")
    print_aggregate(baseline_metrics)
    print("\nCandidate")
    print_aggregate(candidate_metrics)
    print("\nDelta (candidate - baseline)")
    print(f"mean_final_val_bpb_delta={candidate_agg['mean_final_val_bpb'] - baseline_agg['mean_final_val_bpb']:.8f}")
    if "mean_quant_gap" in baseline_agg and "mean_quant_gap" in candidate_agg:
        print(f"mean_quant_gap_delta={candidate_agg['mean_quant_gap'] - baseline_agg['mean_quant_gap']:.8f}")
    if "mean_compressed_bytes" in baseline_agg and "mean_compressed_bytes" in candidate_agg:
        print(
            f"mean_compressed_bytes_delta={candidate_agg['mean_compressed_bytes'] - baseline_agg['mean_compressed_bytes']:.1f}"
        )
    if "mean_tok_s" in baseline_agg and "mean_tok_s" in candidate_agg:
        print(f"mean_tok_s_delta={candidate_agg['mean_tok_s'] - baseline_agg['mean_tok_s']:.1f}")


def run_command(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    overrides = parse_env_overrides(args.set)
    run_experiment(
        repo_root,
        args.label,
        args.seeds,
        overrides,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
    )
    return 0


def summarize_command(args: argparse.Namespace) -> int:
    deduped = resolve_paths(args.paths)
    if not deduped:
        print("No matching log files found.", file=sys.stderr)
        return 1

    metrics_list = [parse_log(path) for path in deduped]
    for metrics in metrics_list:
        print_run_summary(metrics)
    print_aggregate(metrics_list)
    return 0


def compare_command(args: argparse.Namespace) -> int:
    baseline_paths = resolve_paths([args.baseline])
    candidate_paths = resolve_paths([args.candidate])
    if not baseline_paths or not candidate_paths:
        print("Missing baseline or candidate log files.", file=sys.stderr)
        return 1

    baseline_metrics = [parse_log(path) for path in baseline_paths]
    candidate_metrics = [parse_log(path) for path in candidate_paths]
    if aggregate_metrics(baseline_metrics) is None or aggregate_metrics(candidate_metrics) is None:
        print("Need at least one complete run in each set.", file=sys.stderr)
        return 1
    print_compare(baseline_metrics, candidate_metrics)
    return 0


def ladder_command(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    overrides = parse_env_overrides(args.set)
    session_tag = args.session_tag or now_suffix()
    control_label = f"{args.control_label}_{session_tag}"
    screen_label = f"{args.candidate_label}_screen_{session_tag}"
    promoted_label = f"{args.candidate_label}_{session_tag}"

    print(f"Session tag: {session_tag}")
    print(f"Control label: {control_label}")
    print(f"Screen label: {screen_label}")
    print(f"Promoted label: {promoted_label}")

    control_metrics, _ = run_experiment(
        repo_root,
        control_label,
        args.control_seeds,
        {},
        dry_run=args.dry_run,
        keep_going=args.keep_going,
    )
    if args.dry_run:
        print("\nDry run only. No experiments were executed.")
        print(f"Planned compare 1: {control_label} vs {screen_label}")
        print(f"Planned compare 2: {control_label} vs {promoted_label}")
        return 0

    screen_metrics, _ = run_experiment(
        repo_root,
        screen_label,
        args.screen_seeds,
        overrides,
        dry_run=False,
        keep_going=args.keep_going,
    )
    print("\nScreen comparison")
    print_compare(control_metrics, screen_metrics)

    promoted_metrics, _ = run_experiment(
        repo_root,
        promoted_label,
        args.promote_seeds,
        overrides,
        dry_run=False,
        keep_going=args.keep_going,
    )
    print("\nPromoted comparison")
    print_compare(control_metrics, promoted_metrics)
    return 0


def resolve_paths(patterns: list[str]) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(repo_root.glob(pattern))
        if matches:
            files.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and summarize local MLX evaluation experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one experiment label across one or more seeds.")
    run_parser.add_argument("--label", required=True, help="Short experiment label, e.g. control or kv2.")
    run_parser.add_argument("--seeds", nargs="+", type=int, default=[1337], help="Seeds to run.")
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Environment override in KEY=VALUE form. Can be passed multiple times.",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    run_parser.add_argument("--keep-going", action="store_true", help="Continue remaining seeds after a failed run.")
    run_parser.set_defaults(func=run_command)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize one or more existing log files.")
    summarize_parser.add_argument("paths", nargs="+", help="Log file paths or glob patterns relative to the repo root.")
    summarize_parser.set_defaults(func=summarize_command)

    compare_parser = subparsers.add_parser("compare", help="Compare aggregate metrics for two log sets.")
    compare_parser.add_argument("baseline", help="Baseline log glob, e.g. logs/control_s*.txt")
    compare_parser.add_argument("candidate", help="Candidate log glob, e.g. logs/kv2_s*.txt")
    compare_parser.set_defaults(func=compare_command)

    ladder_parser = subparsers.add_parser("ladder", help="Run control, screen, and promoted comparisons automatically.")
    ladder_parser.add_argument("--candidate-label", required=True, help="Candidate label, e.g. kv2.")
    ladder_parser.add_argument("--control-label", default="control", help="Control label prefix.")
    ladder_parser.add_argument("--control-seeds", nargs="+", type=int, default=[1337, 1338, 1339])
    ladder_parser.add_argument("--screen-seeds", nargs="+", type=int, default=[1337])
    ladder_parser.add_argument("--promote-seeds", nargs="+", type=int, default=[1337, 1338, 1339])
    ladder_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Environment override in KEY=VALUE form for the candidate. Can be passed multiple times.",
    )
    ladder_parser.add_argument("--session-tag", help="Optional fixed suffix to keep related labels grouped.")
    ladder_parser.add_argument("--dry-run", action="store_true", help="Print the planned ladder labels without running.")
    ladder_parser.add_argument("--keep-going", action="store_true", help="Continue remaining seeds after a failed run.")
    ladder_parser.set_defaults(func=ladder_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
