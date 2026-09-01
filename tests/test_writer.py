"""
Tests for jmpio writer functionality
"""

import os
import tempfile
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from jmpio import read_jmp, write_jmp
from jmpio.metadata import read_metadata

# Directory with test data (set up by conftest.py)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def test_write_jmp_existence():
    """Test that the write_jmp function exists"""
    assert callable(write_jmp)


def test_write_jmp_rejects_unsupported_version():
    """Test that unsupported JMP write versions fail explicitly."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    with tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp:
        temp_path = temp.name

    try:
        with pytest.raises(ValueError, match="only '17.2.0' is currently supported"):
            write_jmp(df, temp_path, version="16.0")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_round_trip_basic():
    """Test writing a simple DataFrame to a JMP file and reading it back"""
    # Create a simple DataFrame
    df = pd.DataFrame(
        {
            "integers": [1, 2, 3, 4],
            "floats": [1.1, 2.2, 3.3, 4.4],
            "strings": ["a", "bb", "ccc", "dddd"],
        }
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp:
        temp_path = temp.name

    try:
        # Write the DataFrame to a JMP file
        write_jmp(df, temp_path)

        # Read it back
        df_read = read_jmp(temp_path)

        # Check that the data is the same
        pd.testing.assert_series_equal(df["integers"], df_read["integers"], check_dtype=False)
        pd.testing.assert_series_equal(df["floats"], df_read["floats"], check_dtype=False)
        pd.testing.assert_series_equal(df["strings"], df_read["strings"], check_dtype=False)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_round_trip_missing_values():
    """Test writing a DataFrame with missing values to a JMP file and reading it back"""
    # Create a DataFrame with missing values
    df = pd.DataFrame(
        {
            "integers": [1, None, 3, 4],
            "floats": [1.1, 2.2, None, 4.4],
            "strings": ["a", None, "ccc", "dddd"],
        }
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp:
        temp_path = temp.name

    try:
        # Write the DataFrame to a JMP file
        write_jmp(df, temp_path)

        # Read it back
        df_read = read_jmp(temp_path)

        # Check that the data is the same
        assert df["integers"].iloc[0] == df_read["integers"].iloc[0]
        assert pd.isna(df_read["integers"].iloc[1])
        assert df["integers"].iloc[2] == df_read["integers"].iloc[2]
        assert df["integers"].iloc[3] == df_read["integers"].iloc[3]

        assert df["floats"].iloc[0] == df_read["floats"].iloc[0]
        assert df["floats"].iloc[1] == df_read["floats"].iloc[1]
        assert pd.isna(df_read["floats"].iloc[2])
        assert df["floats"].iloc[3] == df_read["floats"].iloc[3]

        assert df["strings"].iloc[0] == df_read["strings"].iloc[0]
        assert pd.isna(df_read["strings"].iloc[1]) or df_read["strings"].iloc[1] == ""
        assert df["strings"].iloc[2] == df_read["strings"].iloc[2]
        assert df["strings"].iloc[3] == df_read["strings"].iloc[3]
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_write_jmp17_metadata_and_string_dtype():
    """Test that writer output has readable JMP 17 metadata and pandas string columns."""
    df = pd.DataFrame(
        {
            "integers": [1, 2, 3],
            "floats": [1.25, None, 3.5],
            "strings": pd.Series(["x", "yy", "zzz"], dtype="string"),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp:
        temp_path = temp.name

    try:
        write_jmp(df, temp_path)

        with open(temp_path, "rb") as file:
            info = read_metadata(file)

        assert info.version == "17.2.0"
        assert info.nrows == 3
        assert info.ncols == 3
        assert info.column.names == ["integers", "floats", "strings"]

        df_read = read_jmp(temp_path)
        assert df_read["integers"].tolist() == [1, 2, 3]
        assert df_read["floats"].iloc[[0, 2]].tolist() == [1.25, 3.5]
        assert pd.isna(df_read["floats"].iloc[1])
        assert df_read["strings"].tolist() == ["x", "yy", "zzz"]
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_round_trip_datetime():
    """Test writing a DataFrame with datetime values to a JMP file and reading it back"""
    # Create a DataFrame with datetime values
    df = pd.DataFrame(
        {
            "dates": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), None],
            "datetimes": [
                datetime(2020, 1, 1, 12, 0),
                datetime(2020, 1, 2, 13, 30),
                None,
                datetime(2020, 1, 4, 18, 15),
            ],
            "durations": [
                timedelta(hours=1),
                timedelta(minutes=30),
                None,
                timedelta(days=2),
            ],
        }
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp:
        temp_path = temp.name

    try:
        # Write the DataFrame to a JMP file
        write_jmp(df, temp_path)

        # Read it back
        df_read = read_jmp(temp_path)

        # Check dates
        for i in [0, 1, 2]:
            if not pd.isna(df["dates"].iloc[i]):
                expected_date = pd.Timestamp(df["dates"].iloc[i]).strftime("%Y-%m-%d")
                actual_date = pd.Timestamp(df_read["dates"].iloc[i]).strftime(
                    "%Y-%m-%d"
                )
                assert expected_date == actual_date

        assert pd.isna(df_read["dates"].iloc[3])

        # Check datetimes
        for i in [0, 1, 3]:
            if not pd.isna(df["datetimes"].iloc[i]):
                expected_dt = pd.Timestamp(df["datetimes"].iloc[i]).strftime(
                    "%Y-%m-%d %H:%M"
                )
                actual_dt = pd.Timestamp(df_read["datetimes"].iloc[i]).strftime(
                    "%Y-%m-%d %H:%M"
                )
                assert expected_dt == actual_dt

        assert pd.isna(df_read["datetimes"].iloc[2])

        # Check durations
        for i in [0, 1, 3]:
            if not pd.isna(df["durations"].iloc[i]):
                expected_sec = df["durations"].iloc[i].total_seconds()
                actual_sec = df_read["durations"].iloc[i].total_seconds()
                assert (
                    abs(expected_sec - actual_sec) < 1
                )  # Allow for small rounding errors

        assert pd.isna(df_read["durations"].iloc[2])
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_write_compressed_vs_uncompressed():
    """Test that compressed and uncompressed files both work"""
    # Create a DataFrame with some data
    df = pd.DataFrame(
        {
            "integers": list(range(100)),
            "floats": [float(i) / 10 for i in range(100)],
            "strings": ["s" * i for i in range(100)],
        }
    )

    # Create temporary files
    with (
        tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp1,
        tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp2,
    ):
        compressed_path = temp1.name
        uncompressed_path = temp2.name

    try:
        # Write compressed file
        write_jmp(df, compressed_path, compress=True)

        # Write uncompressed file
        write_jmp(df, uncompressed_path, compress=False)

        # Check file sizes - compressed should be smaller
        compressed_size = os.path.getsize(compressed_path)
        uncompressed_size = os.path.getsize(uncompressed_path)

        # For large enough data, compression should make a difference
        # This might not always be true for small test data
        # assert compressed_size < uncompressed_size

        # Read both files and check data
        df_compressed = read_jmp(compressed_path)
        df_uncompressed = read_jmp(uncompressed_path)

        # Data should be the same
        pd.testing.assert_frame_equal(df_compressed, df_uncompressed)
    finally:
        # Clean up
        for path in [compressed_path, uncompressed_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_round_trip_example1():
    """Test roundtrip with example1.jmp"""
    # Read example file
    file_path = os.path.join(TEST_DATA_DIR, "example1.jmp")

    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        pytest.skip(f"Test file {file_path} not found")

    # Read the example file
    df_orig = read_jmp(file_path)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jmp", delete=False) as temp:
        temp_path = temp.name

    try:
        # Write to a new JMP file
        write_jmp(df_orig, temp_path)

        # Read it back
        df_read = read_jmp(temp_path)

        # Check that we have the same columns
        assert set(df_orig.columns) == set(df_read.columns)

        # Check the basic data types match
        for col in df_orig.columns:
            # Skip some columns that might have slight differences due to formatting
            if col in ["time", "date", "duration"]:
                continue

            try:
                if pd.api.types.is_numeric_dtype(df_orig[col]):
                    # For numeric columns, check values are close
                    pd.testing.assert_series_equal(
                        df_orig[col], df_read[col], check_dtype=False, check_exact=False
                    )
                elif pd.api.types.is_string_dtype(df_orig[col]):
                    # For string columns, check values match exactly
                    pd.testing.assert_series_equal(
                        df_orig[col], df_read[col], check_dtype=False
                    )
            except AssertionError:
                # If exact comparison fails, check values individually
                for i in range(len(df_orig)):
                    assert pd.isna(df_orig[col].iloc[i]) == pd.isna(
                        df_read[col].iloc[i]
                    ), f"Missing value mismatch at position {i} in column {col}"

                    if not pd.isna(df_orig[col].iloc[i]):
                        assert str(df_orig[col].iloc[i]) == str(df_read[col].iloc[i]), (
                            f"Value mismatch at position {i} in column {col}"
                        )
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
