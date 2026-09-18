# Diarization Error Decomposition Under Pause Annotation Ambiguity

[![arXiv](https://img.shields.io/badge/arXiv-2609.11007-b31b1b.svg)](https://arxiv.org/abs/2609.11007)

`der-decomposition` is the official implementation of our paper
[*Diarization Error Decomposition Under Pause Annotation Ambiguity*](https://arxiv.org/abs/2609.11007).
It extends [`pyannote.metrics`](https://github.com/pyannote/pyannote-metrics)
with an exact decomposition of the standard speaker diarization error rate
(DER):

$$
\mathrm{DER} = \mathrm{Core\ DER} + \mathrm{Pause\ DER}.
$$

The standard DER value and its optimal speaker mapping are unchanged. Missed
detection and false alarm are each divided into pause-attributable and core
components, while speaker confusion is always included in Core DER.

## Installation

From the repository root:

```bash
pip install .
```

Python 3.10 or later and `pyannote.metrics` 4.x are required.

## Python API

```python
from pyannote.core import Annotation, Segment, Timeline

from der_decomposition import DecomposedDiarizationErrorRate


reference = Annotation(uri="example")
reference[Segment(0.0, 1.0)] = "A"
reference[Segment(2.0, 3.0)] = "A"

hypothesis = Annotation(uri="example")
hypothesis[Segment(0.0, 3.0)] = "speaker0"

uem = Timeline([Segment(0.0, 3.0)], uri="example")

metric = DecomposedDiarizationErrorRate(pause_threshold=1.0)
detail = metric(reference, hypothesis, uem=uem, detailed=True)

# Ordinary pyannote DER
print(detail["diarization error rate"])  # 0.5

# Ordinary, core, and pause DER
print(metric.error_rates(detail))
# {
#     "diarization error rate": 0.5,
#     "core diarization error rate": 0.0,
#     "pause diarization error rate": 0.5,
# }
```

DER values returned by the Python API are ratios rather than percentages. For
example, `0.5` means 50%.

### Detailed components

With `detailed=True`, the result contains all standard `pyannote.metrics` DER
components and the following additional durations:

| Key | Meaning | Unit |
| --- | --- | --- |
| `missed detection (pause)` | Missed detection attributable to hypothesis pauses | seconds |
| `missed detection (core)` | Remaining missed detection | seconds |
| `false alarm (pause)` | False alarm attributable to reference pauses | seconds |
| `false alarm (core)` | Remaining false alarm | seconds |

The class also follows the standard `BaseMetric` accumulation API:

```python
metric = DecomposedDiarizationErrorRate()

for reference, hypothesis, uem in dataset:
    metric(reference, hypothesis, uem=uem)

ordinary_der = abs(metric)
core_der = metric.core_der()
pause_der = metric.pause_der()
```

## Evaluation settings

`pause_threshold` specifies the maximum duration of an eligible internal pause.
It defaults to `math.inf`, which places no upper limit on pause duration.
Setting it to zero disables pause attribution.

`collar` and `skip_overlap` follow the definitions used by standard
`pyannote.metrics` DER. Pause eligibility is determined before applying the
reference-boundary collar.

Providing an explicit un-partitioned evaluation map (UEM) is recommended.
Without one, the scoring region is approximated from the union of the reference
and hypothesis extents, so false alarms outside those extents cannot be
measured.

See the accompanying paper for the formal definition of the decomposition.


## Command line

```bash
der-decomposition \
  --reference reference.rttm \
  --hypothesis hypothesis.rttm \
  --uem evaluation.uem
```

The default pause threshold is infinite. Use a finite threshold when needed:

```bash
der-decomposition \
  --reference reference.rttm \
  --hypothesis hypothesis.rttm \
  --uem evaluation.uem \
  --pause-threshold 1.0
```

The command accepts files, directories, multiple paths, and glob patterns.
Use `--skip-overlap` to exclude reference overlap and `--collar` to set a
reference-boundary collar.

Example output:

```text
Reference speaker time : 2.000 s
DER                    : 50.000% (1.000 s)
  core                 : 0.000% (0.000 s)
  pause-attributable   : 50.000% (1.000 s)
Missed detection       : 0.000% (0.000 s)
  core                 : 0.000% (0.000 s)
  pause-attributable   : 0.000% (0.000 s)
False alarm            : 50.000% (1.000 s)
  core                 : 0.000% (0.000 s)
  pause-attributable   : 50.000% (1.000 s)
Confusion              : 0.000% (0.000 s)
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests use synthetic annotations and verify both the exact decomposition
and agreement with standard `pyannote.metrics` DER.

## Citation

```bibtex
@misc{horiguchi2026diarization,
  author        = {Horiguchi, Shota and Delcroix, Marc and Tawara, Naohiro and Plaquet, Alexis},
  title         = {Diarization Error Decomposition Under Pause Annotation Ambiguity},
  year          = {2026},
  month         = sep,
  eprint        = {2609.11007},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.11007},
}
```

## License

Please refer to the [LICENSE](LICENSE) file for details.
