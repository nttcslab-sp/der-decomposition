from __future__ import annotations

import math
import random

import pytest
from der_decomposition import (
    DER_CORE,
    DER_PAUSE,
    FALSE_ALARM_CORE,
    FALSE_ALARM_PAUSE,
    MISS_CORE,
    MISS_PAUSE,
    DecomposedDiarizationErrorRate,
)
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import (
    IER_CONFUSION,
    IER_FALSE_ALARM,
    IER_MISS,
    IER_TOTAL,
)


def annotation(*items: tuple[float, float, str], uri: str = "file") -> Annotation:
    result = Annotation(uri=uri)
    for start, end, speaker in items:
        result[Segment(start, end), speaker] = speaker
    return result


def uem(start: float, end: float, uri: str = "file") -> Timeline:
    return Timeline(segments=[Segment(start, end)], uri=uri)


def assert_standard_components_match(
    reference: Annotation,
    hypothesis: Annotation,
    evaluation_map: Timeline,
    **kwargs,
) -> dict:
    standard = DiarizationErrorRate(**kwargs)(
        reference, hypothesis, uem=evaluation_map, detailed=True
    )
    decomposed = DecomposedDiarizationErrorRate(pause_threshold=1.0, **kwargs)(
        reference, hypothesis, uem=evaluation_map, detailed=True
    )

    for component in (IER_TOTAL, IER_MISS, IER_FALSE_ALARM, IER_CONFUSION):
        assert decomposed[component] == pytest.approx(standard[component])
    assert decomposed["diarization error rate"] == pytest.approx(
        standard["diarization error rate"]
    )
    return decomposed


def test_filled_reference_pause_is_pause_false_alarm() -> None:
    reference = annotation((0.0, 1.0, "A"), (2.0, 3.0, "A"))
    hypothesis = annotation((0.0, 3.0, "X"))
    metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    assert detail[IER_FALSE_ALARM] == pytest.approx(1.0)
    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(1.0)
    assert detail[FALSE_ALARM_CORE] == pytest.approx(0.0)
    rates = metric.error_rates(detail)
    assert rates["diarization error rate"] == pytest.approx(0.5)
    assert rates[DER_CORE] == pytest.approx(0.0)
    assert rates[DER_PAUSE] == pytest.approx(0.5)


def test_filled_hypothesis_pause_is_pause_missed_detection() -> None:
    reference = annotation((0.0, 3.0, "A"))
    hypothesis = annotation((0.0, 1.0, "X"), (2.0, 3.0, "X"))
    metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    assert detail[IER_MISS] == pytest.approx(1.0)
    assert detail[MISS_PAUSE] == pytest.approx(1.0)
    assert detail[MISS_CORE] == pytest.approx(0.0)


def test_isolated_insertion_is_core_error() -> None:
    reference = annotation((0.0, 1.0, "A"), (2.0, 3.0, "A"))
    hypothesis = annotation(
        (0.0, 1.0, "X"),
        (4.0 / 3.0, 5.0 / 3.0, "X"),
        (2.0, 3.0, "X"),
    )
    metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(0.0)
    assert detail[FALSE_ALARM_CORE] == pytest.approx(1.0 / 3.0)


def test_boundary_shift_is_core_error() -> None:
    reference = annotation((0.0, 1.0, "A"), (2.0, 3.0, "A"))
    hypothesis = annotation((0.0, 1.2, "X"), (2.2, 3.0, "X"))
    metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    assert detail[IER_FALSE_ALARM] == pytest.approx(0.2)
    assert detail[IER_MISS] == pytest.approx(0.2)
    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(0.0)
    assert detail[MISS_PAUSE] == pytest.approx(0.0)
    assert detail[FALSE_ALARM_CORE] == pytest.approx(0.2)
    assert detail[MISS_CORE] == pytest.approx(0.2)


def test_pause_longer_than_finite_threshold_is_core_error() -> None:
    reference = annotation((0.0, 1.0, "A"), (2.1, 3.0, "A"))
    hypothesis = annotation((0.0, 3.0, "X"))
    metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(0.0)
    assert detail[FALSE_ALARM_CORE] == pytest.approx(1.1)


def test_default_infinite_threshold_accepts_long_pause() -> None:
    reference = annotation((0.0, 1.0, "A"), (3.0, 4.0, "A"))
    hypothesis = annotation((0.0, 4.0, "X"))
    metric = DecomposedDiarizationErrorRate()

    detail = metric(reference, hypothesis, uem=uem(0.0, 4.0), detailed=True)

    assert math.isinf(metric.pause_threshold)
    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(2.0)
    assert detail[FALSE_ALARM_CORE] == pytest.approx(0.0)


def test_zero_threshold_disables_pause_attribution() -> None:
    reference = annotation((0.0, 1.0, "A"), (2.0, 3.0, "A"))
    hypothesis = annotation((0.0, 3.0, "X"))
    metric = DecomposedDiarizationErrorRate(pause_threshold=0.0)

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(0.0)
    assert detail[FALSE_ALARM_CORE] == pytest.approx(1.0)


def test_zero_overlap_mapping_is_not_a_pause_bridge() -> None:
    reference = annotation((0.0, 1.0, "A"), (2.0, 3.0, "A"))
    hypothesis = annotation((1.0, 2.0, "X"))
    metric = DecomposedDiarizationErrorRate()

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(0.0)
    assert detail[MISS_PAUSE] == pytest.approx(0.0)


def test_collar_changes_scored_amount_but_not_pause_eligibility() -> None:
    reference = annotation((0.0, 1.0, "A"), (2.0, 3.0, "A"))
    hypothesis = annotation((0.0, 3.0, "X"))
    metric = DecomposedDiarizationErrorRate(
        pause_threshold=1.0,
        collar=0.2,
    )

    detail = metric(reference, hypothesis, uem=uem(0.0, 3.0), detailed=True)

    # pyannote collar=0.2 excludes +/-0.1 s around each boundary.
    assert detail[IER_FALSE_ALARM] == pytest.approx(0.8)
    assert detail[FALSE_ALARM_PAUSE] == pytest.approx(0.8)


def test_standard_der_matches_with_overlap_collar_and_uem_holes() -> None:
    reference = annotation(
        (0.0, 3.0, "A"),
        (1.0, 2.5, "B"),
        (3.0, 4.0, "A"),
    )
    hypothesis = annotation(
        (0.0, 1.2, "X"),
        (1.0, 2.0, "Y"),
        (2.0, 4.0, "X"),
        (3.2, 3.7, "Z"),
    )
    evaluation_map = Timeline(
        segments=[Segment(0.2, 2.2), Segment(2.4, 3.8)], uri="file"
    )

    assert_standard_components_match(
        reference,
        hypothesis,
        evaluation_map,
        collar=0.2,
        skip_overlap=False,
    )
    assert_standard_components_match(
        reference,
        hypothesis,
        evaluation_map,
        collar=0.2,
        skip_overlap=True,
    )


def test_components_decompose_exactly() -> None:
    reference = annotation(
        (0.0, 1.0, "A"),
        (2.0, 4.0, "A"),
        (1.5, 3.0, "B"),
    )
    hypothesis = annotation(
        (0.0, 4.0, "X"),
        (1.7, 2.8, "Y"),
        (3.5, 4.0, "Z"),
    )
    metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)
    detail = metric(reference, hypothesis, uem=uem(0.0, 4.0), detailed=True)

    assert detail[IER_MISS] == pytest.approx(detail[MISS_PAUSE] + detail[MISS_CORE])
    assert detail[IER_FALSE_ALARM] == pytest.approx(
        detail[FALSE_ALARM_PAUSE] + detail[FALSE_ALARM_CORE]
    )
    rates = metric.error_rates(detail)
    assert rates["diarization error rate"] == pytest.approx(
        rates[DER_CORE] + rates[DER_PAUSE]
    )


def test_accumulation_uses_pyannote_base_metric_api() -> None:
    metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)
    metric(
        annotation((0.0, 1.0, "A"), (2.0, 3.0, "A"), uri="one"),
        annotation((0.0, 3.0, "X"), uri="one"),
        uem=uem(0.0, 3.0, uri="one"),
    )
    metric(
        annotation((0.0, 2.0, "A"), uri="two"),
        annotation((0.0, 1.0, "X"), uri="two"),
        uem=uem(0.0, 2.0, uri="two"),
    )

    assert metric[IER_TOTAL] == pytest.approx(4.0)
    assert abs(metric) == pytest.approx(0.5)
    assert metric.pause_der() == pytest.approx(0.25)
    assert metric.core_der() == pytest.approx(0.25)


@pytest.mark.parametrize("pause_threshold", [-1.0, math.nan])
def test_rejects_invalid_pause_threshold(pause_threshold: float) -> None:
    with pytest.raises(ValueError):
        DecomposedDiarizationErrorRate(pause_threshold=pause_threshold)


def test_empty_reference_component_rates_are_undefined() -> None:
    reference = annotation()
    hypothesis = annotation((0.0, 1.0, "X"))
    metric = DecomposedDiarizationErrorRate()

    detail = metric(reference, hypothesis, uem=uem(0.0, 1.0), detailed=True)

    assert detail[IER_TOTAL] == 0.0
    assert math.isnan(metric.core_der(detail))
    assert math.isnan(metric.pause_der(detail))


def test_randomized_standard_der_agreement() -> None:
    def random_annotation(rng: random.Random, prefix: str) -> Annotation:
        result = Annotation(uri="random")
        for speaker_index in range(rng.randint(0, 4)):
            speaker = f"{prefix}{speaker_index}"
            active = [rng.random() < 0.4 for _ in range(20)]
            start = None
            for index, value in enumerate(active + [False]):
                if value and start is None:
                    start = 0.2 * index
                elif not value and start is not None:
                    result[Segment(start, 0.2 * index), speaker] = speaker
                    start = None
        return result

    evaluation_map = Timeline(
        segments=[Segment(0.0, 1.8), Segment(2.0, 4.0)], uri="random"
    )
    components = (IER_TOTAL, IER_MISS, IER_FALSE_ALARM, IER_CONFUSION)

    for seed in range(100):
        rng = random.Random(seed)
        reference = random_annotation(rng, "r")
        hypothesis = random_annotation(rng, "h")
        kwargs = {
            "collar": rng.choice([0.0, 0.1, 0.2]),
            "skip_overlap": rng.choice([False, True]),
        }
        standard = DiarizationErrorRate(**kwargs)(
            reference, hypothesis, uem=evaluation_map, detailed=True
        )
        decomposed = DecomposedDiarizationErrorRate(
            pause_threshold=rng.choice([0.0, 0.4, 1.0, math.inf]),
            **kwargs,
        )(
            reference,
            hypothesis,
            uem=evaluation_map,
            detailed=True,
        )

        for component in (*components, "diarization error rate"):
            assert decomposed[component] == pytest.approx(standard[component])
        assert decomposed[IER_MISS] == pytest.approx(
            decomposed[MISS_PAUSE] + decomposed[MISS_CORE]
        )
        assert decomposed[IER_FALSE_ALARM] == pytest.approx(
            decomposed[FALSE_ALARM_PAUSE] + decomposed[FALSE_ALARM_CORE]
        )
