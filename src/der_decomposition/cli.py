"""Command-line interface for DER decomposition."""

from __future__ import annotations

import argparse
import glob
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.identification import (
    IER_CONFUSION,
    IER_FALSE_ALARM,
    IER_MISS,
    IER_TOTAL,
)

from .metric import (
    DER_CORE,
    DER_PAUSE,
    FALSE_ALARM_CORE,
    FALSE_ALARM_PAUSE,
    MISS_CORE,
    MISS_PAUSE,
    DecomposedDiarizationErrorRate,
)


def _expand_paths(inputs: Sequence[Path], suffix: str) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob(f"*{suffix}")))
        elif item.is_file():
            paths.append(item)
        else:
            paths.extend(Path(match) for match in glob.glob(str(item), recursive=True))

    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError(f"no {suffix} files found")
    return unique


def _load_rttm(paths: Sequence[Path]) -> dict[str, Annotation]:
    annotations: dict[str, Annotation] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 8 or fields[0].upper() != "SPEAKER":
                    raise ValueError(f"{path}:{line_number}: invalid RTTM line")
                uri = fields[1]
                start = float(fields[3])
                duration = float(fields[4])
                if duration <= 0.0:
                    continue
                annotation = annotations.setdefault(uri, Annotation(uri=uri))
                # A speaker is a single binary activity stream. Reusing the
                # label as track also prevents overlapping RTTM rows for the
                # same speaker from being counted twice by pyannote.metrics.
                track = fields[7]
                annotation[Segment(start, start + duration), track] = fields[7]
    return annotations


def _load_uem(paths: Sequence[Path]) -> dict[str, Timeline]:
    segments: defaultdict[str, list[Segment]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"{path}:{line_number}: invalid UEM line")
                start = float(fields[2])
                end = float(fields[3])
                if end > start:
                    segments[fields[0]].append(Segment(start, end))
    return {
        uri: Timeline(uri=uri, segments=uri_segments).support()
        for uri, uri_segments in segments.items()
    }


def _percentage(value: float) -> float | None:
    return None if math.isnan(value) else 100.0 * value


def _summary(metric: DecomposedDiarizationErrorRate) -> dict:
    components = metric[:]
    rates = metric.error_rates()
    return {
        "pause_threshold": metric.pause_threshold,
        "collar": metric.collar,
        "skip_overlap": metric.skip_overlap,
        "reference_speaker_time": components[IER_TOTAL],
        "der_percent": _percentage(rates[metric.metric_name()]),
        "core_der_percent": _percentage(rates[DER_CORE]),
        "pause_der_percent": _percentage(rates[DER_PAUSE]),
        "components_seconds": {
            IER_MISS: components[IER_MISS],
            MISS_PAUSE: components[MISS_PAUSE],
            MISS_CORE: components[MISS_CORE],
            IER_FALSE_ALARM: components[IER_FALSE_ALARM],
            FALSE_ALARM_PAUSE: components[FALSE_ALARM_PAUSE],
            FALSE_ALARM_CORE: components[FALSE_ALARM_CORE],
            IER_CONFUSION: components[IER_CONFUSION],
        },
    }


def _print_text(summary: dict) -> None:
    def percent(value: float | None) -> str:
        return "nan" if value is None else f"{value:.3f}%"

    components = summary["components_seconds"]
    total = summary["reference_speaker_time"]

    def component(seconds: float) -> str:
        value = None if total <= 0.0 else 100.0 * seconds / total
        return f"{percent(value)} ({seconds:.3f} s)"

    def rate(value: float | None, seconds: float) -> str:
        return f"{percent(value)} ({seconds:.3f} s)"

    der_seconds = (
        components[IER_MISS] + components[IER_FALSE_ALARM] + components[IER_CONFUSION]
    )
    core_der_seconds = (
        components[MISS_CORE] + components[FALSE_ALARM_CORE] + components[IER_CONFUSION]
    )
    pause_der_seconds = components[MISS_PAUSE] + components[FALSE_ALARM_PAUSE]

    print(f"Reference speaker time : {total:.3f} s")
    print(f"DER                    : {rate(summary['der_percent'], der_seconds)}")
    print(
        f"  core                 : "
        f"{rate(summary['core_der_percent'], core_der_seconds)}"
    )
    print(
        f"  pause-attributable   : "
        f"{rate(summary['pause_der_percent'], pause_der_seconds)}"
    )
    print(f"Missed detection       : {component(components[IER_MISS])}")
    print(f"  core                 : {component(components[MISS_CORE])}")
    print(f"  pause-attributable   : {component(components[MISS_PAUSE])}")
    print(f"False alarm            : {component(components[IER_FALSE_ALARM])}")
    print(f"  core                 : {component(components[FALSE_ALARM_CORE])}")
    print(f"  pause-attributable   : {component(components[FALSE_ALARM_PAUSE])}")
    print(f"Confusion              : {component(components[IER_CONFUSION])}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute standard DER and decompose it into core and pause error."
    )
    parser.add_argument("--reference", "--ref", nargs="+", type=Path, required=True)
    parser.add_argument("--hypothesis", "--hyp", nargs="+", type=Path, required=True)
    parser.add_argument("--uem", "-u", nargs="+", type=Path)
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=math.inf,
        help="Maximum pause duration in seconds (default: inf)",
    )
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--skip-overlap", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        references = _load_rttm(_expand_paths(args.reference, ".rttm"))
        hypotheses = _load_rttm(_expand_paths(args.hypothesis, ".rttm"))
        uems = (
            _load_uem(_expand_paths(args.uem, ".uem")) if args.uem is not None else None
        )

        metric = DecomposedDiarizationErrorRate(
            pause_threshold=args.pause_threshold,
            collar=args.collar,
            skip_overlap=args.skip_overlap,
        )
        uris = sorted(
            uems if uems is not None else references.keys() | hypotheses.keys()
        )
        if not uris:
            raise ValueError("no recordings found")

        for uri in uris:
            reference = references.get(uri, Annotation(uri=uri))
            hypothesis = hypotheses.get(uri, Annotation(uri=uri))
            metric(reference, hypothesis, uem=None if uems is None else uems[uri])

        _print_text(_summary(metric))
        return 0
    except (OSError, ValueError) as error:
        print(f"der-decomposition: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
