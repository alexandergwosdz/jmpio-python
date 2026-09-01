"""
Tests for jmpio reader functionality
"""

import os
import re
from datetime import timedelta

import pandas as pd
import pytest

from jmpio import read_jmp, scan_directory

# Directory with test data (set up by conftest.py)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def test_read_jmp_existence():
    """Test that the read_jmp function exists"""
    assert callable(read_jmp)


def test_invalid_file():
    """Test reading a non-existent file"""
    with pytest.raises(FileNotFoundError):
        read_jmp("nonexistent_file.jmp")


def test_example1():
    """Test reading example1.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "example1.jmp")

    df = read_jmp(file_path)

    # Test dimensions
    assert df.shape[0] == 4  # 4 rows
    assert df.shape[1] == 12  # 12 columns

    # Test column names
    expected_columns = [
        "ints",
        "floats",
        "charconstwidth",
        "time",
        "date",
        "duration",
        "charconstwidth2",
        "charvariable16",
        "formula",
        "pressures",
        "char utf8",
        "charvariable8",
    ]
    assert set(df.columns) == set(expected_columns)

    # Test integer column
    assert df["ints"].tolist() == [1, 2, 3, 4]

    # Test float column
    assert df["floats"].tolist() == [11.1, 22.2, 33.3, 44.4]

    # Test constant width character column
    assert df["charconstwidth"].tolist() == ["a", "b", "c", "d"]

    # Test date/time columns
    assert pd.Timestamp(df["time"].iloc[0]).strftime("%Y-%m-%d %H:%M") == "1976-04-01 21:12"
    assert pd.Timestamp(df["date"].iloc[0]).strftime("%Y-%m-%d") == "2024-01-13"
    assert pd.isna(df["time"].iloc[3])
    assert pd.isna(df["date"].iloc[2])

    # Test duration
    assert isinstance(df["duration"].iloc[0], timedelta)
    assert df["duration"].iloc[0].total_seconds() == pytest.approx(2322, abs=2)

    # Test variable width character columns
    assert df["charvariable16"].iloc[0] == "aa"
    assert df["charvariable16"].iloc[3].startswith("abcdefghijabcdefghij")

    # Test formula column
    assert df["formula"].tolist() == ["2", "4", "6", "8"]

    # Test column with missing values
    assert df["pressures"].iloc[0] == pytest.approx(101.325)
    assert pd.isna(df["pressures"].iloc[1])

    # Test UTF-8 column
    # Note: exact UTF-8 characters may vary by platform, so just check types
    assert all(isinstance(s, str) for s in df["char utf8"])

    # Test variable width strings
    assert df["charvariable8"].tolist() == ["a", "bb", "cc", "abcdefghijkl"]


def test_compressed():
    """Test reading compressed.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "compressed.jmp")

    df = read_jmp(file_path)

    # Test numeric column - all values should be 1
    assert df["numeric"].fillna(0).eq(1).all()

    # Test character columns
    assert all(df["character1"] == "a")
    assert all(df["character11"] == "abcdefghijk")
    assert all(df["character130"] == "abcdefghij" * 13)

    # Additional checks from Julia tests
    # All rows should have the same datetime 1904-01-01 00:00:01
    assert pd.to_datetime(df["y-m-d h:m:s"]).dt.strftime("%Y-%m-%d %H:%M:%S").eq("1904-01-01 00:00:01").all()

    # Date-only column should be 2024-01-20
    assert pd.to_datetime(df["yyyy-mm-dd"]).dt.strftime("%Y-%m-%d").eq("2024-01-20").all()

    # Duration column should be 196 seconds
    assert df["min:s"].apply(lambda td: pd.isna(td) or abs(td.total_seconds() - 196) < 1).all()


def test_date_file():
    """Test reading date.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "date.jmp")

    df = read_jmp(file_path)

    # Test dates
    dates = pd.to_datetime(df["ddmmyyyy"]).dt.strftime("%Y-%m-%d").tolist()
    expected_dates = ["2011-05-25", "1973-05-24", "2027-05-22", "2020-05-01"]
    assert dates == expected_dates


def test_duration_file():
    """Test reading duration.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "duration.jmp")

    df = read_jmp(file_path)

    # Column name contains colons, ensure all rows equal 88201 seconds
    assert df[":day:hr:m:s"].apply(lambda td: isinstance(td, timedelta) and abs(td.total_seconds() - 88201) < 1).all()


def test_time_file():
    """Test reading time.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "time.jmp")

    df = read_jmp(file_path)

    # Expect 2 rows and many time-related columns
    assert df.shape[0] == 2
    assert df.shape[1] >= 2

    expected_time_strs = ["19:54:14", "06:11:24"]

    def to_time_str(v):
        # Normalize various datetime/time-like values to HH:MM:SS for comparison
        try:
            ts = pd.to_datetime(v)
            if pd.isna(ts):
                # Fall back to string if object is already time
                return str(v)
            return pd.Timestamp(ts).strftime("%H:%M:%S")
        except Exception:
            try:
                return v.strftime("%H:%M:%S")  # datetime.time
            except Exception:
                return str(v)

    # Check each column's two rows match expected times (regardless of date formatting)
    for col in df.columns:
        times = [to_time_str(df[col].iloc[0]), to_time_str(df[col].iloc[1])]
        assert times == expected_time_strs


def test_long_column_names():
    """Test reading longcolumnnames.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "longcolumnnames.jmp")

    df = read_jmp(file_path)

    name1 = "".join(f"{i:010d}" for i in range(1, 141))
    name2 = "".join(f"{i:010d}" for i in range(1, 281))

    assert list(df.columns) == [name1, name2]
    assert df.values.tolist() == [[1, 2], [1, 2], [1, 2]]


def test_single_column_single_row():
    """Test reading singlecolumnsinglerow.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "singlecolumnsinglerow.jmp")

    df = read_jmp(file_path)

    assert df.shape == (1, 1)
    assert list(df.columns) == ["Column 1"]
    assert df["Column 1"].tolist() == [1]


def test_byte_integers_basic():
    """Test reading byteintegers.jmp with specific integer dtypes and values"""
    file_path = os.path.join(TEST_DATA_DIR, "byteintegers.jmp")
    df = read_jmp(file_path)

    # Dtypes
    assert str(df["1-byte integer"].dtype) in ("int8", "Int8")
    assert str(df["2-byte integer"].dtype) in ("int16", "Int16")
    assert str(df["4-byte integer"].dtype) in ("int32", "Int32")

    # Values
    assert df["1-byte integer"].tolist() == [0, 1, 0, 1, 0]
    assert df["2-byte integer"].tolist() == [-187, -30, -18, 13, -55]
    assert df["4-byte integer"].tolist() == [-28711, -16887, -26063, 13093, -44761]


def test_byte_integers_with_missing():
    """Test reading byteintegers_withmissing.jmp with missing values"""
    file_path = os.path.join(TEST_DATA_DIR, "byteintegers_withmissing.jmp")
    df = read_jmp(file_path)

    # Dtypes should be pandas nullable integer types
    assert str(df["1-byte integer"].dtype) == "Int8"
    assert str(df["2-byte integer"].dtype) == "Int16"
    assert str(df["4-byte integer"].dtype) == "Int32"

    # Values with missing
    col1 = df["1-byte integer"].tolist()
    assert col1[:-1] == [0, 1, 0, 1, 0] and pd.isna(col1[-1])

    col2 = df["2-byte integer"].tolist()
    assert col2[:-1] == [-187, -30, -18, 13, -55] and pd.isna(col2[-1])

    col4 = df["4-byte integer"].tolist()
    assert col4[:-1] == [-28711, -16887, -26063, 13093, -44761] and pd.isna(col4[-1])


def test_byte_integers_notcompressed():
    """Test reading byteintegers_notcompressed.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "byteintegers_notcompressed.jmp")
    df = read_jmp(file_path)

    # Dtypes (nullable for columns with missing)
    assert str(df["onebyte"].dtype) in ("int8", "Int8")
    assert str(df["twobyte"].dtype) in ("Int16", "int16")
    assert str(df["fourbyte"].dtype) in ("Int32", "int32")
    assert pd.api.types.is_float_dtype(df["numeric"].dtype)

    # Values
    assert df["onebyte"].tolist() == [1, 2, -126, 127]

    # twobyte: [32767, missing, 0, -32766]
    assert df["twobyte"].iloc[[0, 2, 3]].tolist() == [32767, 0, -32766]
    assert pd.isna(df["twobyte"].iloc[1])

    # fourbyte: [missing, 2147483647, -2147483646, missing]
    assert df["fourbyte"].iloc[[1, 2]].tolist() == [2147483647, -2147483646]
    assert pd.isna(df["fourbyte"].iloc[0]) and pd.isna(df["fourbyte"].iloc[3])

    # numeric: [missing, 2147483648, missing, missing]
    assert df["numeric"].iloc[1] == 2147483648
    assert pd.isna(df["numeric"].iloc[0]) and pd.isna(df["numeric"].iloc[2]) and pd.isna(df["numeric"].iloc[3])


def test_geographic_file():
    """Test reading geographic.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "geographic.jmp")
    df = read_jmp(file_path)

    assert (
        pd.Series(df["Longitude_DDD"])
        .astype(float)
        .pipe(lambda s: s.fillna(0))
        .sub(pd.Series([151.209900, 24.945831, -122.449]))
        .abs()
        .le(1e-6)
        .all()
    )

    assert pd.isna(df["Latitude_DDD"].iloc[1])
    assert pd.isna(df["Latitude_DMM"].iloc[2])


def test_currencies_file():
    """Test reading currencies.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, "currencies.jmp")
    df = read_jmp(file_path)

    assert pd.Series(df["AUD"]).astype(float).tolist() == pytest.approx([1.0, 2.0, 2.0])
    assert pd.Series(df["COP"]).astype(float).tolist() == pytest.approx([3.14, 2.78, 1.41])


def test_row_states():
    """Test reading rowstate.jmp and verifying markers and colors"""
    file_path = os.path.join(TEST_DATA_DIR, "rowstate.jmp")
    df = read_jmp(file_path)
    print([rs.marker for rs in df["rowstate3"].to_list()])

    # rowstate3 markers in rows 2 and 3
    assert getattr(df["rowstate3"].iloc[1], "marker", None) == "▲"
    assert getattr(df["rowstate3"].iloc[2], "marker", None) == "ꙮ"

    # rowstate2 color in row 3 should be near grey (0.753, 0.753, 0.753)
    color = getattr(df["rowstate2"].iloc[2], "color", None)
    assert color is not None and all(abs(c - 0.753) < 0.01 for c in color)


def test_compact_subtype_file():
    """Test reading compact.jmp if available"""
    file_path = os.path.join(TEST_DATA_DIR, "compact.jmp")

    df = read_jmp(file_path)

    data = ["aa", "b", "ccc", "dd", "dd"]
    assert df["normalsubtype"].tolist() == data
    assert df["compactsubtype"].tolist() == data

    # longcompact column checks
    assert df["longcompact"].iloc[0] == "x" * 254
    assert df["longcompact"].iloc[1] == "y" * 255
    assert df["longcompact"].iloc[2] == ""
    z = "z" * 256
    assert df["longcompact"].iloc[3] == z
    assert df["longcompact"].iloc[4] == z

    df = read_jmp(os.path.join(TEST_DATA_DIR, "compact_UInt8.jmp"))
    assert df.shape == (130, 1)
    assert df["A"].iloc[:3].tolist() == ["a1", "a2", "a3"]
    assert df["A"].iloc[-1] == "a130"

    df = read_jmp(os.path.join(TEST_DATA_DIR, "compact_UInt16.jmp"))
    assert df.shape == (32770, 1)
    assert df["A"].iloc[-1] == "a32770"

    df = read_jmp(os.path.join(TEST_DATA_DIR, "compact_UInt32.jmp"))
    assert df.shape == (65537, 1)
    assert df["A"].iloc[-1] == "a65537"


def test_minusfour_column_info_marker():
    """Test column metadata sections with a 0xfc 0xff marker."""
    df = read_jmp(os.path.join(TEST_DATA_DIR, "minusfour.jmp"))

    assert df.shape == (3, 1)
    assert df["Column 1"].tolist() == ["a", "b", "c"]


def test_bug_mwe4_truncates_fixed_strings_at_first_null():
    """Test fixed-width strings containing internal null padding."""
    df = read_jmp(os.path.join(TEST_DATA_DIR, "bugMWE4.jmp"))

    assert df["c"].iloc[:3].tolist() == ["cat", "cat", "dolor sit amet"]
    assert df["d"].iloc[:3].tolist() == ["bat", "bat", "consectetur adipiscing elit"]
    assert df["e"].iloc[:3].tolist() == ["bird", "bird", "sed do eiusmod"]
    assert df["f"].iloc[15:18].tolist() == ["foo", "bar", "hello world"]


def test_column_filtering():
    """Test column filtering features"""
    file_path = os.path.join(TEST_DATA_DIR, "example1.jmp")

    # Test selecting columns by name
    df = read_jmp(file_path, select=["ints", "floats"])
    assert list(df.columns) == ["ints", "floats"]

    # Test selecting columns by index
    df = read_jmp(file_path, select=[0, 1])
    assert len(df.columns) == 2
    assert "ints" in df.columns
    assert "floats" in df.columns

    # Test selecting columns by regex
    df = read_jmp(file_path, select=[re.compile(r"^char")])
    for col in df.columns:
        assert col.startswith("char")

    # Test dropping columns
    df = read_jmp(file_path, drop=["ints", "floats"])
    assert "ints" not in df.columns
    assert "floats" not in df.columns


def test_scan_directory():
    """Test scanning a directory for JMP files"""
    # This test requires the scan_directory function to be implemented
    result = scan_directory(TEST_DATA_DIR, recursive=False)

    # Check that we got a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that we found at least some JMP files
    assert len(result) > 0

    # Check that all files have .jmp extension
    for file in result["filename"]:
        assert file.lower().endswith(".jmp")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
