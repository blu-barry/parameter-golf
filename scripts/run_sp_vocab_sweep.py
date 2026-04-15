#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "experiment_runs"
DEFAULT_GENERATED_TOKENIZER_CONFIG = ROOT / "data" / "tokenizer_specs_sp12288_sp16384.json"
KEVCLARK_REPO_ID = "kevclark/parameter-golf"
DEFAULT_DOCS_REPO_ID = "willdepueoai/parameter-golf"

STEP_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+train_time:(?P<train_time_ms>[0-9.]+)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>[0-9.]+)ms\s+artifact_bytes:(?P<artifact_bytes>\d+)"
)
FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
STOP_RE = re.compile(
    r"stopping_early:\s+wallclock_cap\s+train_time:(?P<train_time_ms>[0-9.]+)ms\s+step:(?P<step>\d+)/(?P<iterations>\d+)"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")
TOKENIZER_SETUP_RE = re.compile(
    r"tokenizer_setup:vocab_size:(?P<vocab_size>\d+)\s+embedding_table_shape:\[(?P<rows>\d+),(?P<cols>\d+)\]"
)


@dataclass
class SweepResult:
    variant: str
    run_id: str
    status: str
    returncode: int
    elapsed_wallclock_s: float
    steps_completed: int | None
    iterations_target: int | None
    latest_val_loss: float | None
    latest_val_bpb: float | None
    final_val_loss: float | None
    final_val_bpb: float | None
    final_val_loss_exact: float | None
    final_val_bpb_exact: float | None
    artifact_bytes: int | None
    artifact_mb_decimal: float | None
    eval_time_ms: float | None
    train_time_ms: float | None
    model_params: int | None
    vocab_size: int | None
    embedding_rows: int | None
    embedding_cols: int | None
    data_path: str
    tokenizer_path: str
    log_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential SentencePiece vocab sweep on one GPU.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["sp4096", "sp8192", "sp12288", "sp16384"],
        help="Tokenizer variants to compare, e.g. sp4096 sp8192 sp12288.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for run logs and summary files.",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=10,
        help="Training shards to download for published variants like sp4096 and sp8192.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing dataset/tokenizer artifacts before each run.",
    )
    parser.add_argument(
        "--prepare-artifacts",
        action="store_true",
        help="Prepare the full 4-way sweep: download sp4096/sp8192 and locally generate sp12288/sp16384.",
    )
    parser.add_argument(
        "--generated-train-shards",
        type=int,
        default=10,
        help="When locally generating sp12288/sp16384, export this many train shards worth of tokens.",
    )
    parser.add_argument(
        "--tokenizer-train-docs",
        type=int,
        default=None,
        help="Optional doc cap for training the local SentencePiece tokenizers.",
    )
    parser.add_argument(
        "--generated-tokenizer-config",
        type=Path,
        default=DEFAULT_GENERATED_TOKENIZER_CONFIG,
        help="Tokenizer config JSON used to build local variants such as sp12288 and sp16384.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for helper commands and training.",
    )
    parser.add_argument(
        "--torchrun",
        default="torchrun",
        help="torchrun executable name or path.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Processes per node. Use 1 for a single H100 sweep.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variables to pass to every run.",
    )
    parser.add_argument(
        "--run-prefix",
        default="sp_sweep",
        help="Prefix for generated RUN_ID values.",
    )
    return parser.parse_args()


def variant_to_vocab_size(variant: str) -> int:
    if not variant.startswith("sp") or not variant[2:].isdigit():
        raise ValueError(f"Expected variant like sp8192, got {variant!r}")
    return int(variant[2:])


def variant_to_paths(variant: str) -> tuple[Path, Path]:
    vocab_size = variant_to_vocab_size(variant)
    data_path = ROOT / "data" / "datasets" / f"fineweb10B_{variant}"
    tokenizer_path = ROOT / "data" / "tokenizers" / f"fineweb_{vocab_size}_bpe.model"
    return data_path, tokenizer_path


def parse_key_values(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def ensure_variant_downloaded(args: argparse.Namespace, variant: str) -> None:
    data_path, tokenizer_path = variant_to_paths(variant)
    if data_path.exists() and tokenizer_path.exists():
        return
    cmd = [
        args.python,
        str(ROOT / "data" / "cached_challenge_fineweb.py"),
        "--variant",
        variant,
        "--train-shards",
        str(args.train_shards),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_published_variants(args: argparse.Namespace, variants: list[str]) -> None:
    if not variants:
        return
    manifest_path = ROOT / "data" / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["MATCHED_FINEWEB_REPO_ID"] = KEVCLARK_REPO_ID
    for variant in variants:
        data_path, tokenizer_path = variant_to_paths(variant)
        if data_path.exists() and tokenizer_path.exists():
            continue
        cmd = [
            args.python,
            str(ROOT / "data" / "cached_challenge_fineweb.py"),
            "--variant",
            variant,
            "--train-shards",
            str(args.train_shards),
        ]
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def prepare_generated_variants(args: argparse.Namespace, variants: list[str]) -> None:
    if not variants:
        return
    if all(variant_to_paths(variant)[0].exists() and variant_to_paths(variant)[1].exists() for variant in variants):
        return
    cmd = [
        args.python,
        str(ROOT / "data" / "download_hf_docs_and_tokenize.py"),
        "--repo-id",
        DEFAULT_DOCS_REPO_ID,
        "--remote-root",
        "datasets",
        "--output-root",
        str(ROOT / "data"),
        "--tokenizer-config",
        str(args.generated_tokenizer_config),
        "--skip-byte",
        "--max-train-shards",
        str(args.generated_train_shards),
    ]
    if args.tokenizer_train_docs is not None:
        cmd.extend(["--tokenizer-train-docs", str(args.tokenizer_train_docs)])
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_artifacts(args: argparse.Namespace) -> None:
    variants = list(dict.fromkeys(args.variants))
    published_variants = [variant for variant in variants if variant in {"sp4096", "sp8192"}]
    generated_variants = [variant for variant in variants if variant in {"sp12288", "sp16384"}]
    unsupported = [variant for variant in variants if variant not in {"sp4096", "sp8192", "sp12288", "sp16384"}]
    if unsupported:
        raise ValueError(f"--prepare-artifacts does not know how to build {unsupported!r}")
    prepare_published_variants(args, published_variants)
    prepare_generated_variants(args, generated_variants)


def stream_command(command: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def parse_log(log_path: Path, variant: str, run_id: str, returncode: int, elapsed_wallclock_s: float) -> SweepResult:
    latest_val_loss = None
    latest_val_bpb = None
    final_val_loss = None
    final_val_bpb = None
    final_val_loss_exact = None
    final_val_bpb_exact = None
    artifact_bytes = None
    eval_time_ms = None
    train_time_ms = None
    steps_completed = None
    iterations_target = None
    model_params = None
    vocab_size = None
    embedding_rows = None
    embedding_cols = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if match := STEP_VAL_RE.search(line):
            latest_val_loss = float(match.group("val_loss"))
            latest_val_bpb = float(match.group("val_bpb"))
            steps_completed = int(match.group("step"))
            iterations_target = int(match.group("iterations"))
            train_time_ms = float(match.group("train_time_ms"))
        if match := FINAL_RE.search(line):
            final_val_loss = float(match.group("val_loss"))
            final_val_bpb = float(match.group("val_bpb"))
            eval_time_ms = float(match.group("eval_time_ms"))
            artifact_bytes = int(match.group("artifact_bytes"))
        if match := FINAL_EXACT_RE.search(line):
            final_val_loss_exact = float(match.group("val_loss"))
            final_val_bpb_exact = float(match.group("val_bpb"))
        if match := STOP_RE.search(line):
            steps_completed = int(match.group("step"))
            iterations_target = int(match.group("iterations"))
            train_time_ms = float(match.group("train_time_ms"))
        if match := MODEL_PARAMS_RE.search(line):
            model_params = int(match.group("model_params"))
        if match := TOKENIZER_SETUP_RE.search(line):
            vocab_size = int(match.group("vocab_size"))
            embedding_rows = int(match.group("rows"))
            embedding_cols = int(match.group("cols"))

    data_path, tokenizer_path = variant_to_paths(variant)
    artifact_mb_decimal = artifact_bytes / 1_000_000 if artifact_bytes is not None else None
    status = "ok" if returncode == 0 and final_val_bpb_exact is not None else "failed"
    return SweepResult(
        variant=variant,
        run_id=run_id,
        status=status,
        returncode=returncode,
        elapsed_wallclock_s=elapsed_wallclock_s,
        steps_completed=steps_completed,
        iterations_target=iterations_target,
        latest_val_loss=latest_val_loss,
        latest_val_bpb=latest_val_bpb,
        final_val_loss=final_val_loss,
        final_val_bpb=final_val_bpb,
        final_val_loss_exact=final_val_loss_exact,
        final_val_bpb_exact=final_val_bpb_exact,
        artifact_bytes=artifact_bytes,
        artifact_mb_decimal=artifact_mb_decimal,
        eval_time_ms=eval_time_ms,
        train_time_ms=train_time_ms,
        model_params=model_params,
        vocab_size=vocab_size,
        embedding_rows=embedding_rows,
        embedding_cols=embedding_cols,
        data_path=str(data_path),
        tokenizer_path=str(tokenizer_path),
        log_path=str(log_path),
    )


def write_csv(results: list[SweepResult], path: Path) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(results: list[SweepResult], path: Path) -> None:
    payload = [asdict(result) for result in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(results: list[SweepResult], path: Path) -> None:
    lines = [
        "# SP Vocab Sweep Results",
        "",
        "| variant | status | final_val_bpb | artifact_mb | steps_completed | train_time_ms | eval_time_ms | log |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.variant,
                    result.status,
                    fmt(result.final_val_bpb_exact),
                    fmt(result.artifact_mb_decimal),
                    fmt(result.steps_completed),
                    fmt(result.train_time_ms),
                    fmt(result.eval_time_ms),
                    result.log_path,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `final_val_bpb` is taken from `final_int8_zlib_roundtrip_exact` when present.",
            "- Runs are sequential, so they are directly comparable on one GPU under one software environment.",
            "- Keep architecture and optimizer env vars fixed across variants if you want this to isolate tokenizer effects.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_artifacts:
        prepare_artifacts(args)

    extra_env = parse_key_values(args.set)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    sweep_dir = args.output_dir / f"{timestamp}_{args.run_prefix}"
    logs_dir = sweep_dir / "logs"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[SweepResult] = []

    for index, variant in enumerate(args.variants, start=1):
        if args.download_missing:
            print(f"[setup] ensuring data exists for {variant}", flush=True)
            ensure_variant_downloaded(args, variant)

        vocab_size = variant_to_vocab_size(variant)
        data_path, tokenizer_path = variant_to_paths(variant)
        run_id = f"{args.run_prefix}_{index:02d}_{variant}_{timestamp}"
        log_path = logs_dir / f"{run_id}.log"

        env = os.environ.copy()
        env.update(extra_env)
        env.update(
            {
                "RUN_ID": run_id,
                "DATA_PATH": str(data_path),
                "TOKENIZER_PATH": str(tokenizer_path),
                "VOCAB_SIZE": str(vocab_size),
            }
        )

        command = [
            args.torchrun,
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            "train_gpt.py",
        ]

        print(f"\n=== [{index}/{len(args.variants)}] variant={variant} run_id={run_id} ===", flush=True)
        print(f"data_path={data_path}", flush=True)
        print(f"tokenizer_path={tokenizer_path}", flush=True)
        start = time.perf_counter()
        returncode = stream_command(command, env=env, log_path=log_path)
        elapsed_wallclock_s = time.perf_counter() - start

        result = parse_log(
            log_path=log_path,
            variant=variant,
            run_id=run_id,
            returncode=returncode,
            elapsed_wallclock_s=elapsed_wallclock_s,
        )
        results.append(result)

        print(
            f"[done] variant={variant} status={result.status} "
            f"final_val_bpb={fmt(result.final_val_bpb_exact)} "
            f"artifact_mb={fmt(result.artifact_mb_decimal)} "
            f"steps={fmt(result.steps_completed)}",
            flush=True,
        )

    csv_path = sweep_dir / "summary.csv"
    json_path = sweep_dir / "summary.json"
    md_path = sweep_dir / "README.md"
    write_csv(results, csv_path)
    write_json(results, json_path)
    write_markdown(results, md_path)

    print("\n=== sweep summary ===")
    for result in results:
        print(
            f"{result.variant}: status={result.status} "
            f"final_val_bpb={fmt(result.final_val_bpb_exact)} "
            f"artifact_mb={fmt(result.artifact_mb_decimal)} "
            f"steps={fmt(result.steps_completed)} "
            f"log={result.log_path}"
        )
    print(f"summary_csv={csv_path}")
    print(f"summary_json={json_path}")
    print(f"summary_md={md_path}")


if __name__ == "__main__":
    main()
