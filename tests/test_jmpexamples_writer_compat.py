"""
Writer compatibility checks against native JMP files.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from jmpio import read_jmp, write_jmp
from jmpio.constants import GZIP_SECTION_START
from jmpio.metadata import read_metadata


JMPEXAMPLES_DIR = Path(os.environ.get("JMPIO_JMPEXAMPLES_DIR", r"C:\GitHub\jmpexamples"))
REPO_TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

KNOWN_METADATA_LIMITATIONS = {
    "08Apr14 - BG00002-J expiry extension ANCOVA analyses.jmp",
    "09Apr14 - IMPD update DS file.jmp",
    "18Apr14 - IMPD update DP file.jmp",
    "Comparison of SC to IV - DS.jmp",
    "TCQ DS data for TE allowance.jmp",
    "TCQ DS.jmp",
    "TSCIM DS 25°C.jmp",
    "Tys High Concentration DP 25C.jmp",
    "Tys SCIM 25C Purity.jmp",
}


def _jmp_files(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    return tuple(sorted(root.rglob("*.jmp"), key=lambda path: str(path).lower()))


JMPEXAMPLE_FILES = _jmp_files(JMPEXAMPLES_DIR)


def _read_info(path: Path):
    with path.open("rb") as file:
        return read_metadata(file)


def _column_body_start(data: bytes, offset: int) -> int:
    name_len = int.from_bytes(data[offset : offset + 2], "little", signed=True)
    assert 0 < name_len < 32768, f"invalid column name length {name_len} at offset {offset}"
    return offset + 2 + name_len


def _column_signature(data: bytes, offset: int) -> bytes:
    body_start = _column_body_start(data, offset)
    return data[body_start : body_start + 6]


def _assert_native_column_property_block(data: bytes, offset: int, column_name: str) -> None:
    body_start = _column_body_start(data, offset)
    gzip_pos = data.find(GZIP_SECTION_START, body_start, min(len(data), body_start + 512))
    search_end = gzip_pos if gzip_pos != -1 else min(len(data), body_start + 128)

    for pos in range(body_start + 6, max(body_start + 6, search_end - 3)):
        property_count = int.from_bytes(data[pos : pos + 2], "little")
        property_bytes = int.from_bytes(data[pos + 2 : pos + 4], "little")
        if 1 <= property_count <= 32 and 8 <= property_bytes <= 4096:
            return

    snippet = data[body_start : min(len(data), search_end + 16)].hex(" ")
    assert False, (
        f"{column_name!r} does not have a native-looking column property record "
        f"between byte {body_start + 6} and {search_end}: {snippet}"
    )


@lru_cache(maxsize=1)
def _native_column_signatures() -> frozenset[bytes]:
    signatures = set()
    for root in (JMPEXAMPLES_DIR, REPO_TEST_DATA_DIR):
        for path in _jmp_files(root):
            try:
                info = _read_info(path)
            except Exception:
                continue

            data = path.read_bytes()
            for offset in info.column.offsets:
                signatures.add(_column_signature(data, offset))

    assert signatures, "no native JMP column signatures were discovered"
    return frozenset(signatures)


def _xfail_known_metadata_limitation(path: Path) -> None:
    if path.name in KNOWN_METADATA_LIMITATIONS:
        pytest.xfail("current reader metadata scanner does not handle this native file yet")


def _require_jmpexamples() -> tuple[Path, ...]:
    if not JMPEXAMPLE_FILES:
        pytest.skip(f"JMP examples directory not found: {JMPEXAMPLES_DIR}")
    return JMPEXAMPLE_FILES


def test_jmpexamples_corpus_is_available():
    """Ensure the external native JMP corpus is being exercised."""
    files = _require_jmpexamples()
    assert len(files) >= 80


@pytest.mark.parametrize("path", JMPEXAMPLE_FILES, ids=lambda path: path.name)
def test_jmpexamples_metadata_is_supported(path: Path):
    """Verify metadata parsing across all known native JMP examples."""
    _xfail_known_metadata_limitation(path)

    info = _read_info(path)

    assert info.nrows >= 0
    assert info.ncols == len(info.column.names)
    assert info.ncols == len(info.column.offsets)
    assert all(offset > 0 for offset in info.column.offsets)


@pytest.mark.parametrize("path", JMPEXAMPLE_FILES, ids=lambda path: path.name)
def test_jmpexamples_native_columns_have_property_blocks(path: Path):
    """Verify native examples contain column metadata before payload bytes."""
    _xfail_known_metadata_limitation(path)

    data = path.read_bytes()
    info = _read_info(path)
    native_signatures = _native_column_signatures()

    for column_name, offset in zip(info.column.names, info.column.offsets):
        assert _column_signature(data, offset) in native_signatures
        _assert_native_column_property_block(data, offset, column_name)


@pytest.mark.parametrize("path", JMPEXAMPLE_FILES, ids=lambda path: path.name)
@pytest.mark.xfail(
    strict=True,
    reason="writer still omits native JMP per-column object/property records",
)
def test_writer_roundtrip_from_jmpexamples_has_native_column_blocks(path: Path):
    """Writer output from native examples should preserve native column structure."""
    _xfail_known_metadata_limitation(path)

    df = read_jmp(str(path))
    native_signatures = _native_column_signatures()

    with tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        write_jmp(df, str(temp_path))
        data = temp_path.read_bytes()
        info = _read_info(temp_path)

        assert info.nrows == len(df)
        assert info.ncols == len(df.columns)
        assert info.column.names == list(df.columns)

        for column_name, offset in zip(info.column.names, info.column.offsets):
            assert _column_signature(data, offset) in native_signatures
            _assert_native_column_property_block(data, offset, column_name)
    finally:
        if temp_path.exists():
            temp_path.unlink()
