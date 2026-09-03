"""
Tests for optional JMP JSL automation helpers.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

import jmpio.writer as writer_module
from jmpio import read_jmp, write_jmp
from jmpio.jsl import (
    JSLAutomationError,
    _build_validate_jsl,
    _jsl_string,
    _parse_status,
    find_jmp_executable,
    write_jmp_with_jsl,
)


def test_find_jmp_executable_prefers_explicit_path(tmp_path):
    executable = tmp_path / "jmp.exe"
    executable.write_text("", encoding="utf-8")

    assert find_jmp_executable(executable) == str(executable)


def test_jsl_string_normalizes_paths_and_quotes():
    assert _jsl_string(r'C:\Temp\has "quote".jmp') == '"C:/Temp/has \\"quote\\".jmp"'


def test_parse_status_preserves_column_names_with_commas(tmp_path):
    result = _parse_status(
        "OK\n154\n2\nLot Number\n1,2-dihydro-RTA 408\n",
        tmp_path / "file.jmp",
        tmp_path / "status.txt",
        tmp_path / "script.jsl",
    )

    assert result.nrows == 154
    assert result.ncols == 2
    assert result.column_names == ["Lot Number", "1,2-dihydro-RTA 408"]


def test_parse_status_raises_on_jmp_error(tmp_path):
    with pytest.raises(JSLAutomationError, match="Cannot open table"):
        _parse_status(
            "ERROR\nCannot open table\n",
            tmp_path / "file.jmp",
            tmp_path / "status.txt",
            tmp_path / "script.jsl",
        )


def test_build_validate_jsl_uses_line_based_status(tmp_path):
    script = _build_validate_jsl(tmp_path / "file.jmp", tmp_path / "status.txt")

    assert 'status = "OK\\!N" || Char( N Rows( dt ) )' in script
    assert 'status ||= "\\!N" || names[i];' in script


def test_write_jmp_jsl_engine_dispatches(monkeypatch, tmp_path):
    calls = []

    def fake_write_jmp_with_jsl(df, filename, *, jmp_executable=None, timeout=60):
        calls.append((df, filename, jmp_executable, timeout))
        Path(filename).write_bytes(b"native")

    monkeypatch.setattr(writer_module, "write_jmp_with_jsl", fake_write_jmp_with_jsl)

    output_path = tmp_path / "out.jmp"
    df = pd.DataFrame({"x": [1]})
    write_jmp(
        df,
        str(output_path),
        engine="jsl",
        jmp_executable="jmp.exe",
        jsl_timeout=5,
    )

    assert output_path.read_bytes() == b"native"
    assert calls == [(df, str(output_path), "jmp.exe", 5)]


def test_write_jmp_auto_falls_back_after_jmp_validation_failure(monkeypatch, tmp_path):
    calls = []

    def fake_write_python(df, filename, *, compress=True, version="17.2.0"):
        calls.append(("python", filename, compress, version))
        Path(filename).write_bytes(b"python")

    def fake_validate(filename, *, jmp_executable=None, timeout=60):
        calls.append(("validate", filename, jmp_executable, timeout))
        raise JSLAutomationError("invalid")

    def fake_write_jmp_with_jsl(df, filename, *, jmp_executable=None, timeout=60):
        calls.append(("jsl", filename, jmp_executable, timeout))
        Path(filename).write_bytes(b"native")

    monkeypatch.setattr(writer_module, "_write_jmp_python", fake_write_python)
    monkeypatch.setattr(writer_module, "validate_jmp_with_jsl", fake_validate)
    monkeypatch.setattr(writer_module, "write_jmp_with_jsl", fake_write_jmp_with_jsl)

    output_path = tmp_path / "out.jmp"
    write_jmp(
        pd.DataFrame({"x": [1]}),
        str(output_path),
        engine="auto",
        jmp_executable="jmp.exe",
        jsl_timeout=5,
    )

    assert output_path.read_bytes() == b"native"
    assert [call[0] for call in calls] == ["python", "validate", "jsl"]


def test_write_jmp_rejects_unknown_engine(tmp_path):
    with pytest.raises(ValueError, match="engine must be one of"):
        write_jmp(pd.DataFrame({"x": [1]}), str(tmp_path / "out.jmp"), engine="bogus")


@pytest.mark.skipif(
    os.environ.get("JMPIO_RUN_JSL_TESTS") != "1" or find_jmp_executable() is None,
    reason="set JMPIO_RUN_JSL_TESTS=1 and install JMP to run JSL automation smoke tests",
)
def test_write_jmp_with_jsl_smoke(tmp_path):
    output_path = tmp_path / "native.jmp"
    df = pd.DataFrame({"lot": ["A", "B"], "value": [1.25, 2.5]})

    result = write_jmp_with_jsl(df, output_path, timeout=30)

    assert result.nrows == 2
    assert result.ncols == 2
    assert result.column_names == ["lot", "value"]
    pd.testing.assert_frame_equal(read_jmp(str(output_path)), df)
