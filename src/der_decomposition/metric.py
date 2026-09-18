"""Pause-aware decomposition of diarization error rate."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import pairwise

from pyannote.core import Annotation, Timeline
from pyannote.core.utils.generators import int_generator, string_generator
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import (
    IER_CONFUSION,
    IER_FALSE_ALARM,
    IER_MISS,
    IER_TOTAL,
    IdentificationErrorRate,
)
from pyannote.metrics.types import Details, MetricComponents

EPSILON = 1e-10
Interval = tuple[float, float]
SpeakerIntervals = dict[str, list[Interval]]

MISS_PAUSE = "missed detection (pause)"
MISS_CORE = "missed detection (core)"
FALSE_ALARM_PAUSE = "false alarm (pause)"
FALSE_ALARM_CORE = "false alarm (core)"

DER_PAUSE = "diarization error rate (pause)"
DER_CORE = "diarization error rate (core)"


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(
        (float(start), float(end)) for start, end in intervals if end - start > EPSILON
    )
    if not ordered:
        return []

    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1] + EPSILON:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _intersect_intervals(
    first: Sequence[Interval], second: Sequence[Interval]
) -> list[Interval]:
    intersections: list[Interval] = []
    i = 0
    j = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end - start > EPSILON:
            intersections.append((start, end))
        if first[i][1] < second[j][1] - EPSILON:
            i += 1
        else:
            j += 1
    return intersections


def _timeline_intervals(timeline: Timeline) -> list[Interval]:
    return [(segment.start, segment.end) for segment in timeline.support()]


def _annotation_intervals(annotation: Annotation) -> SpeakerIntervals:
    return {
        label: [
            (segment.start, segment.end)
            for segment in annotation.label_timeline(label).support()
        ]
        for label in annotation.labels()
    }


def _internal_gaps_with_flanks(
    intervals: Sequence[Interval],
    uem: Sequence[Interval],
    threshold: float,
) -> list[tuple[Interval, Interval, Interval]]:
    """Return eligible-length internal gaps and their speech flanks."""
    if threshold <= 0.0:
        return []

    merged = _merge_intervals(intervals)
    gaps: list[tuple[Interval, Interval, Interval]] = []
    for uem_start, uem_end in _merge_intervals(uem):
        local = _intersect_intervals(merged, [(uem_start, uem_end)])
        for left, right in pairwise(local):
            gap = (left[1], right[0])
            duration = gap[1] - gap[0]
            if duration > EPSILON and duration <= threshold + EPSILON:
                gaps.append((gap, left, right))
    return gaps


def _one_sided_bridge_gaps(
    source_intervals: Sequence[Interval],
    bridge_intervals: Sequence[Interval],
    uem: Sequence[Interval],
    threshold: float,
) -> list[Interval]:
    """Find short internal source gaps continuously bridged by the other side."""
    if threshold <= 0.0:
        return []

    bridge = _merge_intervals(bridge_intervals)
    if not bridge:
        return []
    bridge_starts = [start for start, _ in bridge]

    eligible: list[Interval] = []
    for gap, _, _ in _internal_gaps_with_flanks(source_intervals, uem, threshold):
        gap_start, gap_end = gap

        # Touching either boundary is not enough: the same bridge interval
        # must extend into speech on both sides.
        index = bisect_right(bridge_starts, gap_start - EPSILON) - 1
        if index < 0:
            continue
        bridge_start, bridge_end = bridge[index]
        if bridge_start < gap_start - EPSILON and bridge_end > gap_end + EPSILON:
            eligible.append(gap)

    return eligible


def _eligible_gaps(
    reference: Mapping[str, Sequence[Interval]],
    hypothesis: Mapping[str, Sequence[Interval]],
    mapping: Mapping[str, str],
    uem: Sequence[Interval],
    threshold: float,
) -> tuple[SpeakerIntervals, SpeakerIntervals]:
    reference_gaps: defaultdict[str, list[Interval]] = defaultdict(list)
    hypothesis_gaps: defaultdict[str, list[Interval]] = defaultdict(list)

    for hypothesis_speaker, reference_speaker in mapping.items():
        ref_spans = reference.get(reference_speaker, [])
        hyp_spans = hypothesis.get(hypothesis_speaker, [])

        reference_gaps[reference_speaker].extend(
            _one_sided_bridge_gaps(ref_spans, hyp_spans, uem, threshold)
        )
        hypothesis_gaps[hypothesis_speaker].extend(
            _one_sided_bridge_gaps(hyp_spans, ref_spans, uem, threshold)
        )

    return (
        {
            speaker: _merge_intervals(intervals)
            for speaker, intervals in reference_gaps.items()
            if intervals
        },
        {
            speaker: _merge_intervals(intervals)
            for speaker, intervals in hypothesis_gaps.items()
            if intervals
        },
    )


def _add_interval_events(
    events: defaultdict[float, defaultdict[tuple[str, str], int]],
    intervals: Mapping[str, Sequence[Interval]],
    kind: str,
) -> None:
    for speaker, spans in intervals.items():
        for start, end in spans:
            events[start][(kind, speaker)] += 1
            events[end][(kind, speaker)] -= 1


def _iter_regions(
    uem: Sequence[Interval],
    reference: Mapping[str, Sequence[Interval]],
    hypothesis: Mapping[str, Sequence[Interval]],
    reference_gaps: Mapping[str, Sequence[Interval]],
    hypothesis_gaps: Mapping[str, Sequence[Interval]],
) -> Iterator[tuple[float, set[str], set[str], set[str], set[str]]]:
    """Sweep all activity boundaries once and yield constant-label regions."""
    scoring_uem = _merge_intervals(uem)
    if not scoring_uem:
        return

    events: defaultdict[float, defaultdict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for start, end in scoring_uem:
        events[start][("score", "")] += 1
        events[end][("score", "")] -= 1

    _add_interval_events(events, reference, "reference")
    _add_interval_events(events, hypothesis, "hypothesis")
    _add_interval_events(events, reference_gaps, "reference_gap")
    _add_interval_events(events, hypothesis_gaps, "hypothesis_gap")

    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    previous: float | None = None

    for time in sorted(events):
        if (
            previous is not None
            and time - previous > EPSILON
            and counts.get(("score", ""), 0) > 0
        ):
            yield (
                time - previous,
                {
                    speaker
                    for (kind, speaker), count in counts.items()
                    if kind == "reference" and count > 0
                },
                {
                    speaker
                    for (kind, speaker), count in counts.items()
                    if kind == "hypothesis" and count > 0
                },
                {
                    speaker
                    for (kind, speaker), count in counts.items()
                    if kind == "reference_gap" and count > 0
                },
                {
                    speaker
                    for (kind, speaker), count in counts.items()
                    if kind == "hypothesis_gap" and count > 0
                },
            )

        for key, delta in events[time].items():
            counts[key] += delta
            if counts[key] == 0:
                del counts[key]
        previous = time


def _pause_components(
    scoring_uem: Sequence[Interval],
    reference: SpeakerIntervals,
    hypothesis: SpeakerIntervals,
    mapping: Mapping[str, str],
    reference_gaps: SpeakerIntervals,
    hypothesis_gaps: SpeakerIntervals,
) -> tuple[float, float]:
    inverse_mapping = {
        reference_speaker: hypothesis_speaker
        for hypothesis_speaker, reference_speaker in mapping.items()
    }
    miss_pause = 0.0
    false_alarm_pause = 0.0

    for duration, ref_active, hyp_active, ref_gaps, hyp_gaps in _iter_regions(
        scoring_uem,
        reference,
        hypothesis,
        reference_gaps,
        hypothesis_gaps,
    ):
        mapped_active = {
            hypothesis_speaker: mapping[hypothesis_speaker]
            for hypothesis_speaker in hyp_active
            if hypothesis_speaker in mapping
        }
        mapped_reference = set(mapped_active.values())

        miss_count = max(0, len(ref_active) - len(hyp_active))
        false_alarm_count = max(0, len(hyp_active) - len(ref_active))

        false_alarm_candidates = sum(
            1
            for mapped_ref in mapped_active.values()
            if mapped_ref not in ref_active and mapped_ref in ref_gaps
        )
        false_alarm_pause += duration * min(false_alarm_count, false_alarm_candidates)

        miss_candidates = 0
        for reference_speaker in ref_active - mapped_reference:
            hypothesis_speaker = inverse_mapping.get(reference_speaker)
            if hypothesis_speaker is not None and hypothesis_speaker in hyp_gaps:
                miss_candidates += 1
        miss_pause += duration * min(miss_count, miss_candidates)

    return miss_pause, false_alarm_pause


def _safe_normalized(numerator: float, denominator: float) -> float:
    if denominator <= EPSILON:
        return math.nan
    return numerator / denominator


class DecomposedDiarizationErrorRate(DiarizationErrorRate):
    """Standard DER with pause-attributable/core error decomposition.

    The returned metric value is the ordinary ``pyannote.metrics`` DER. Only
    missed detection and false alarm are split; speaker confusion is always a
    core error. Therefore, for non-empty reference speaker time,

    ``DER = core DER + pause DER``.

    Parameters
    ----------
    pause_threshold : float, optional
        Maximum duration, in seconds, of an internal pause eligible for the
        pause-attributable component. Defaults to ``math.inf``.
    collar : float, optional
        Collar duration interpreted exactly as in
        :class:`pyannote.metrics.diarization.DiarizationErrorRate`.
    skip_overlap : bool, optional
        Whether to exclude reference overlap from evaluation.

    Notes
    -----
    Pause eligibility is determined before applying the reference-boundary
    collar. Mapping and all scored components are computed after applying the
    collar and ``skip_overlap`` setting, as in standard pyannote DER.
    """

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            *super().metric_components(),
            MISS_PAUSE,
            MISS_CORE,
            FALSE_ALARM_PAUSE,
            FALSE_ALARM_CORE,
        ]

    def __init__(
        self,
        pause_threshold: float = math.inf,
        collar: float = 0.0,
        skip_overlap: bool = False,
        **kwargs,
    ) -> None:
        if math.isnan(pause_threshold) or pause_threshold < 0.0:
            raise ValueError("pause_threshold must be non-negative")

        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.pause_threshold = pause_threshold

    @staticmethod
    def _pyannote_label_mappings(
        reference: Annotation,
        hypothesis: Annotation,
        scored_reference: Annotation,
        scored_hypothesis: Annotation,
    ) -> tuple[dict, dict]:
        """Reproduce pyannote DER label normalization on scored speakers.

        Speakers that only occur outside the scoring UEM are assigned later
        generator values. They can participate in gap geometry but never in
        the optimal mapping.
        """
        reference_generator = string_generator()
        hypothesis_generator = int_generator()
        reference_mapping = {
            label: next(reference_generator) for label in scored_reference.labels()
        }
        hypothesis_mapping = {
            label: next(hypothesis_generator) for label in scored_hypothesis.labels()
        }
        for label in reference.labels():
            if label not in reference_mapping:
                reference_mapping[label] = next(reference_generator)
        for label in hypothesis.labels():
            if label not in hypothesis_mapping:
                hypothesis_mapping[label] = next(hypothesis_generator)
        return reference_mapping, hypothesis_mapping

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
        # Resolve the implicit UEM once so the usual pyannote warning is emitted
        # only once, then reuse the same base evaluation map below.
        if uem is None:
            _, _, base_uem = self.uemify(
                reference,
                hypothesis,
                uem=None,
                collar=0.0,
                skip_overlap=False,
                returns_uem=True,
            )
        else:
            base_uem = uem

        scored_reference, scored_hypothesis, scoring_uem = self.uemify(
            reference,
            hypothesis,
            uem=base_uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        # Keep collar holes out of gap geometry. Reference overlap exclusions,
        # when requested, remain genuine UEM holes and cannot form a pause.
        gap_reference, gap_hypothesis, gap_uem = self.uemify(
            reference,
            hypothesis,
            uem=base_uem,
            collar=0.0,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        ref_labels, hyp_labels = self._pyannote_label_mappings(
            reference,
            hypothesis,
            scored_reference,
            scored_hypothesis,
        )
        scored_reference = scored_reference.rename_labels(mapping=ref_labels)
        scored_hypothesis = scored_hypothesis.rename_labels(mapping=hyp_labels)
        gap_reference = gap_reference.rename_labels(mapping=ref_labels)
        gap_hypothesis = gap_hypothesis.rename_labels(mapping=hyp_labels)

        mapping = self.mapper_(scored_hypothesis, scored_reference)
        mapped_hypothesis = scored_hypothesis.rename_labels(mapping=mapping)

        # Delegate the ordinary DER components to pyannote.metrics itself. This
        # makes the main score exactly the standard DER by construction.
        detail = IdentificationErrorRate.compute_components(
            self,
            scored_reference,
            mapped_hypothesis,
            uem=scoring_uem,
            collar=0.0,
            skip_overlap=False,
            **kwargs,
        )

        reference_intervals = _annotation_intervals(gap_reference)
        hypothesis_intervals = _annotation_intervals(gap_hypothesis)
        gap_uem_intervals = _timeline_intervals(gap_uem)
        scoring_uem_intervals = _timeline_intervals(scoring_uem)

        reference_gaps, hypothesis_gaps = _eligible_gaps(
            reference_intervals,
            hypothesis_intervals,
            mapping,
            gap_uem_intervals,
            self.pause_threshold,
        )
        miss_pause, false_alarm_pause = _pause_components(
            scoring_uem_intervals,
            reference_intervals,
            hypothesis_intervals,
            mapping,
            reference_gaps,
            hypothesis_gaps,
        )

        # Numerical noise at shared boundaries can only be sub-EPSILON. A
        # larger excess indicates an internal inconsistency and should surface.
        if miss_pause > detail[IER_MISS] + 1e-7:
            raise RuntimeError("pause missed detection exceeds total missed detection")
        if false_alarm_pause > detail[IER_FALSE_ALARM] + 1e-7:
            raise RuntimeError("pause false alarm exceeds total false alarm")

        miss_pause = min(miss_pause, detail[IER_MISS])
        false_alarm_pause = min(false_alarm_pause, detail[IER_FALSE_ALARM])
        detail[MISS_PAUSE] = miss_pause
        detail[MISS_CORE] = detail[IER_MISS] - miss_pause
        detail[FALSE_ALARM_PAUSE] = false_alarm_pause
        detail[FALSE_ALARM_CORE] = detail[IER_FALSE_ALARM] - false_alarm_pause
        return detail

    def core_der(self, detail: Details | None = None) -> float:
        """Return core DER for one detailed result or accumulated results."""
        detail = self.accumulated_ if detail is None else detail
        numerator = detail[MISS_CORE] + detail[FALSE_ALARM_CORE] + detail[IER_CONFUSION]
        return _safe_normalized(numerator, detail[IER_TOTAL])

    def pause_der(self, detail: Details | None = None) -> float:
        """Return pause DER for one detailed result or accumulated results."""
        detail = self.accumulated_ if detail is None else detail
        numerator = detail[MISS_PAUSE] + detail[FALSE_ALARM_PAUSE]
        return _safe_normalized(numerator, detail[IER_TOTAL])

    def error_rates(self, detail: Details | None = None) -> dict[str, float]:
        """Return ordinary, core, and pause DER from the same components."""
        detail = self.accumulated_ if detail is None else detail
        return {
            self.metric_name(): self.compute_metric(detail),
            DER_CORE: self.core_der(detail),
            DER_PAUSE: self.pause_der(detail),
        }
