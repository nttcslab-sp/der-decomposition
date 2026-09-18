from __future__ import annotations

import math

from der_decomposition.cli import _parser, main


def test_cli_text_output(tmp_path, monkeypatch, capsys) -> None:
    reference = tmp_path / "reference.rttm"
    hypothesis = tmp_path / "hypothesis.rttm"
    evaluation_map = tmp_path / "evaluation.uem"

    reference.write_text(
        "SPEAKER file 1 0 1 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER file 1 2 1 <NA> <NA> A <NA> <NA>\n",
        encoding="utf-8",
    )
    hypothesis.write_text(
        "SPEAKER file 1 0 3 <NA> <NA> X <NA> <NA>\n",
        encoding="utf-8",
    )
    evaluation_map.write_text(
        "file 1 0 3\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "der-decomposition",
            "--reference",
            str(reference),
            "--hypothesis",
            str(hypothesis),
            "--uem",
            str(evaluation_map),
        ],
    )

    assert main() == 0

    output = capsys.readouterr().out
    assert output.splitlines() == [
        "Reference speaker time : 2.000 s",
        "DER                    : 50.000% (1.000 s)",
        "  core                 : 0.000% (0.000 s)",
        "  pause-attributable   : 50.000% (1.000 s)",
        "Missed detection       : 0.000% (0.000 s)",
        "  core                 : 0.000% (0.000 s)",
        "  pause-attributable   : 0.000% (0.000 s)",
        "False alarm            : 50.000% (1.000 s)",
        "  core                 : 0.000% (0.000 s)",
        "  pause-attributable   : 50.000% (1.000 s)",
        "Confusion              : 0.000% (0.000 s)",
    ]


def test_cli_uses_infinite_pause_threshold_by_default() -> None:
    args = _parser().parse_args(
        [
            "--reference",
            "reference.rttm",
            "--hypothesis",
            "hypothesis.rttm",
        ]
    )

    assert math.isinf(args.pause_threshold)