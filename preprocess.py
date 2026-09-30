"""Convert STEP files into the NPZ samples and splits consumed by train.py."""

import argparse
import json
import math
import os
import random
from collections import Counter
from concurrent.futures import TimeoutError, as_completed
from pathlib import Path

from pebble import ProcessExpired, ProcessPool
from tqdm import tqdm


def discover_steps(source: Path) -> list[Path]:
    candidates = [source] if source.is_file() else source.rglob("*")
    files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in {".step", ".stp"})
    if not files:
        raise ValueError(f"No STEP/STP files found in {source}")
    seen = {}
    for path in files:
        key = path.stem.casefold()
        if key in seen:
            raise ValueError(f"Duplicate input stem: {seen[key]} and {path}; use unique filenames")
        seen[key] = path
    return files


def read_source_splits(path: Path, files: list[Path]) -> dict[str, str]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict) or "train" not in raw:
        raise ValueError("Source splits must be an object containing a train list")
    prefixes = {}
    for split, values in raw.items():
        split = "val" if split == "validation" else split
        if split not in {"train", "val", "test"} or not isinstance(values, list):
            raise ValueError("Source splits must contain train/val/test lists")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Split identifiers must be nonempty strings")
            prefix = Path(value.replace("\\", "/")).name
            if Path(prefix).suffix.lower() in {".step", ".stp", ".npz"}:
                prefix = Path(prefix).stem
            if prefix in prefixes and prefixes[prefix] != split:
                raise ValueError(f"Identifier {prefix!r} appears in multiple splits")
            prefixes[prefix] = split
    result = {}
    # Match prefix lengths rather than scanning every identifier for each input.
    lengths = sorted({len(prefix) for prefix in prefixes})
    groups = {}
    for file in files:
        matches = {prefixes[file.stem[:n]] for n in lengths if file.stem[:n] in prefixes}
        if len(matches) > 1:
            raise ValueError(f"Overlapping source prefixes assign {file.name} to multiple splits")
        if matches:
            split = matches.pop()
            group = file.stem[:33]
            if group in groups and groups[group] != split:
                raise ValueError(f"Related source model {group!r} appears in multiple splits")
            groups[group] = split
            result[file.name] = split
    if not result:
        raise ValueError("No input STEP files match the source split identifiers")
    return result


def build_splits(results, source_splits, val_ratio, test_ratio, seed):
    accepted = [row for row in results if row["status"] in {"written", "existing"}]
    splits = {"train": [], "val": [], "test": []}
    if source_splits is None:
        groups = sorted({Path(row["source"]).stem[:33] for row in accepted})
        random.Random(seed).shuffle(groups)
        n_val, n_test = int(len(groups) * val_ratio), int(len(groups) * test_ratio)
        assignment = {group: "train" for group in groups}
        assignment.update({group: "val" for group in groups[:n_val]})
        assignment.update({group: "test" for group in groups[n_val : n_val + n_test]})
        source_splits = {
            row["source"]: assignment[Path(row["source"]).stem[:33]] for row in accepted
        }
    for row in accepted:
        # Full filenames also work with train.py's prefix matcher and avoid
        # accidental matches between IDs such as "part1" and "part10".
        splits[source_splits[row["source"]]].append(row["output"])
    return {split: sorted(names) for split, names in splits.items()}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="STEP/STP file or directory (recursive)"
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("data/npz"))
    parser.add_argument(
        "-w", "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 1) - 2))
    )
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per input STEP file")
    parser.add_argument("--max-faces", type=int, default=50)
    parser.add_argument("--max-edges", type=int, default=1000)
    parser.add_argument("--max-face-edges", type=int, default=30)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Shape-splitting tolerance in STEP import units",
    )
    parser.add_argument(
        "--split-file", type=Path, help="Preserve an existing source train/val/test split"
    )
    parser.add_argument("--splits-output", type=Path, help="Default: OUTPUT/../splits.json")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZ samples")
    args = parser.parse_args()
    if min(args.workers, args.max_faces, args.max_edges, args.max_face_edges) < 1:
        parser.error("Workers and geometry limits must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in (args.timeout, args.tolerance)):
        parser.error("Timeout and tolerance must be finite and positive")
    if (
        any(not math.isfinite(x) or not 0 <= x < 1 for x in (args.val_ratio, args.test_ratio))
        or args.val_ratio + args.test_ratio >= 1
    ):
        parser.error("Split ratios must be nonnegative and sum to less than one")
    splits_path = args.splits_output or args.output.parent / "splits.json"
    report_path = args.output / "preprocessing.json"
    if splits_path.resolve() == report_path.resolve():
        parser.error("Split output and preprocessing report must be different files")
    if args.split_file and args.split_file.resolve() == splits_path.resolve():
        parser.error("Use different paths for the source split file and generated splits")
    try:
        from flowar.preprocessing.step import PreprocessOptions, convert_file

        files = discover_steps(args.input)
        source_splits = read_source_splits(args.split_file, files) if args.split_file else None
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if source_splits is not None:
        files = [file for file in files if file.name in source_splits]
    options = PreprocessOptions(args.max_faces, args.max_edges, args.max_face_edges, args.tolerance)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with tqdm(total=len(files), desc="STEP to NPZ") as progress:
        # Bound the number of queued futures when processing many files.
        with ProcessPool(max_workers=args.workers, max_tasks=100) as pool:
            batch_size = args.workers * 16
            for offset in range(0, len(files), batch_size):
                futures = {
                    pool.schedule(
                        convert_file,
                        args=(str(file), str(args.output), options, args.overwrite),
                        timeout=args.timeout,
                    ): file
                    for file in files[offset : offset + batch_size]
                }
                for future in as_completed(futures):
                    file = futures[future]
                    try:
                        results.extend(future.result())
                    except (TimeoutError, ProcessExpired) as exc:
                        results.append(
                            {"source": file.name, "status": "failed", "reason": type(exc).__name__}
                        )
                    except Exception as exc:
                        results.append(
                            {"source": file.name, "status": "failed", "reason": str(exc)}
                        )
                    progress.update(1)
    results.sort(key=lambda row: (row["source"], row.get("solid", -1)))
    splits = build_splits(results, source_splits, args.val_ratio, args.test_ratio, args.seed)
    counts = dict(Counter(row["status"] for row in results))
    write_json(report_path, {"counts": counts, "results": results})
    print("Conversion:", ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Report: {report_path}")
    if not splits["train"]:
        parser.exit(1, "No accepted training samples; split output was not updated.\n")
    write_json(splits_path, splits)
    print("Splits:", ", ".join(f"{key}={len(value)}" for key, value in splits.items()))
    print(f"Split file: {splits_path}")


if __name__ == "__main__":
    main()
