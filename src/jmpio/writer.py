"""
Functions for writing JMP files
"""

import gzip
import os
import struct
from datetime import date, datetime, time, timedelta
from typing import BinaryIO

import numpy as np
import pandas as pd

from .constants import (
    GZIP_SECTION_START,
    JMP_STARTDATE,
    MAGIC_JMP,
    ROWSTATE_COLORS,
    ROWSTATE_MARKERS,
)
from .jsl import JSLAutomationError, validate_jmp_with_jsl, write_jmp_with_jsl
from .types import RowState

SUPPORTED_WRITE_VERSION = "17.2.0"
JMP17_METADATA_OFFSET = 368
JMP17_PREAMBLE_AFTER_MAGIC = bytes.fromhex(
    """
    12 00 00 00 00 00 00 00 00 00 06 00 06 00 00 00
    75 74 66 2d 38 00 00 00 07 57 61 72 6e 69 6e 67
    20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20
    20 20 20 20 20 20 20 20 02 02 00 00 43 00 01 00
    01 00 08 00 00 00 00 00 02 00 00 00 00 00 37 54
    68 69 73 20 64 61 74 61 20 74 61 62 6c 65 20 68
    61 64 20 62 65 65 6e 20 73 61 76 65 64 20 69 6e
    20 61 20 6d 6f 72 65 20 72 65 63 65 6e 74 20 66
    6f 72 6d 61 74 2e 00 00 00 00 00 00 00 00 00 00
    00 42 49 74 20 69 73 20 6e 6f 74 20 63 6f 6d 70
    61 74 69 62 6c 65 20 77 69 74 68 20 74 68 65 20
    66 6f 72 6d 61 74 20 6b 6e 6f 77 6e 20 74 6f 20
    74 68 69 73 20 76 65 72 73 69 6f 6e 20 6f 66 20
    4a 4d 50 2e 1d 50 6c 65 61 73 65 20 64 6f 20 6e
    6f 74 20 73 61 76 65 20 74 68 69 73 20 74 61 62
    6c 65 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    00 00 00 00 00 00 00 41 53 61 76 69 6e 67 20 74
    68 69 73 20 74 61 62 6c 65 20 77 69 74 68 20 74
    68 69 73 20 76 65 72 73 69 6f 6e 20 6f 66 20 4a
    4d 50 20 77 69 6c 6c 20 64 65 73 74 72 6f 79 20
    74 68 65 20 64 61 74 61 2e 00 07 00 00 00
    """
)

if len(MAGIC_JMP) + len(JMP17_PREAMBLE_AFTER_MAGIC) != JMP17_METADATA_OFFSET:
    raise RuntimeError("JMP17 preamble length does not align with metadata offset")


def _column_visibility_record_len(n_visible: int, n_hidden: int) -> int:
    """Length field used by JMP 17 before visible/hidden column metadata."""
    return 18 + 4 * (n_visible + n_hidden) + 2 * (n_visible + n_hidden)


def _write_compressed_column_prefix(file: BinaryIO, property_count: int = 3) -> None:
    """
    Write native JMP 17 per-column object records before compressed payload.
    """
    if property_count == 4:
        file.write(
            bytes.fromhex(
                """
                00 00 00 00 00 00 00 00 00 00 00 00 00 00
                04 00 20 00 00 00 00 00 02 00 00 00 00 00
                0b 00 02 00 00 00 40 00
                17 00 02 00 00 00 02 00
                20 00 02 00 00 00 42 00
                02 01 00 00
                """
            )
        )
        return

    file.write(
        bytes.fromhex(
            """
            00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            03 00 18 00 00 00 00 00 02 00 00 00 00 00
            17 00 02 00 00 00 00 00
            20 00 02 00 00 00 2d 00
            01 01 00 00
            """
        )
    )


def _gzip_compress(data: bytes) -> bytes:
    """Return deterministic gzip bytes like native JMP fixtures."""
    return gzip.compress(data, mtime=0)


def write_jmp(
    df: pd.DataFrame,
    filename: str,
    compress: bool = True,
    version: str = SUPPORTED_WRITE_VERSION,
    engine: str = "python",
    jmp_executable: str | os.PathLike[str] | None = None,
    jsl_timeout: float = 60,
) -> None:
    """
    Write a pandas DataFrame to a JMP file

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to write
    filename : str
        Path to the output file
    compress : bool, default=True
        Whether to compress the data
    version : str, default="17.2.0"
        JMP file version to write with the Python writer. Currently only
        "17.2.0" is supported.
    engine : {"python", "jsl", "auto"}, default="python"
        Writer engine. "python" uses jmpio's native binary writer. "jsl" uses
        a local JMP installation through generated JSL scripts. "auto" writes
        with the Python writer and falls back to JSL if JMP validation fails.
    jmp_executable : str or path-like, optional
        Path to JMP executable for "jsl" or "auto" engines. If omitted,
        JMPIO_JMP_EXE and common install locations are searched.
    jsl_timeout : float, default=60
        Seconds to wait for each JMP automation script.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> import pandas as pd
    >>> import jmpio
    >>>
    >>> # Create a DataFrame
    >>> df = pd.DataFrame({
    >>>     'ints': [1, 2, 3, 4],
    >>>     'floats': [1.1, 2.2, 3.3, 4.4],
    >>>     'strings': ['a', 'bb', 'ccc', 'dddd']
    >>> })
    >>>
    >>> # Write to a JMP file
    >>> jmpio.write_jmp(df, 'output.jmp')
    """
    engine = engine.lower()
    if engine not in {"python", "jsl", "auto"}:
        raise ValueError("engine must be one of 'python', 'jsl', or 'auto'")

    if engine == "jsl":
        write_jmp_with_jsl(
            df,
            filename,
            jmp_executable=jmp_executable,
            timeout=jsl_timeout,
        )
        return

    if version != SUPPORTED_WRITE_VERSION:
        raise ValueError(
            f"Unsupported JMP write version {version!r}; "
            f"only {SUPPORTED_WRITE_VERSION!r} is currently supported"
        )

    _write_jmp_python(df, filename, compress=compress, version=version)

    if engine == "auto":
        try:
            validate_jmp_with_jsl(
                filename,
                jmp_executable=jmp_executable,
                timeout=jsl_timeout,
            )
        except FileNotFoundError:
            return
        except JSLAutomationError:
            write_jmp_with_jsl(
                df,
                filename,
                jmp_executable=jmp_executable,
                timeout=jsl_timeout,
            )


def _write_jmp_python(
    df: pd.DataFrame,
    filename: str,
    compress: bool = True,
    version: str = SUPPORTED_WRITE_VERSION,
) -> None:
    """Write a pandas DataFrame using jmpio's native binary writer."""
    # Create directory if it doesn't exist
    directory = os.path.dirname(os.path.abspath(filename))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Open file in binary write mode
    with open(filename, "wb") as file:
        # Write file header
        write_file_header(file, df, version)

        # Write column metadata
        offset_table_pos = write_column_metadata(file, df)

        # Write column data
        column_offsets = []
        for i, column_name in enumerate(df.columns):
            column_offsets.append(file.tell())
            column_data = df[column_name]
            write_column_data(file, column_data, column_offsets[i], column_name, compress)

        end_pos = file.tell()
        file.seek(offset_table_pos)
        for offset in column_offsets:
            file.write(struct.pack("<q", offset))
        file.seek(end_pos)

        # Any final corrections or clean-up
        finalize_file(file)


def write_file_header(file: BinaryIO, df: pd.DataFrame, version: str) -> None:
    """
    Write JMP file header

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    df : pd.DataFrame
        DataFrame being written
    version : str
        JMP version to use in the header
    """
    # JMP 17 places table metadata at byte 368 after a compatibility preamble.
    # Native JMP-authored files in the fixture set share this preamble for
    # ordinary data tables.
    file.write(MAGIC_JMP)
    file.write(JMP17_PREAMBLE_AFTER_MAGIC)

    # Write number of rows (Int64) and columns (Int32)
    file.write(struct.pack("<q", len(df)))
    file.write(struct.pack("<i", len(df.columns)))

    # Unknown values observed in JMP 17.2 files. The last value is the length
    # marker for the following charset string.
    file.write(struct.pack("<5h", 0, 4, 93, 0, 6))

    # JMP strings in this section use a 4-byte little-endian length prefix.
    charset = b"utf-8\0"
    file.write(struct.pack("<i", len(charset)))
    file.write(charset)

    # The reader searches for these bytes: 07 00 08 00 00 00.
    file.write(struct.pack("<3H", 7, 8, 0))

    # Write save time (current time as seconds since JMP epoch)
    current_time = datetime.now()
    seconds_since_epoch = (current_time - JMP_STARTDATE).total_seconds()
    file.write(struct.pack("<d", seconds_since_epoch))

    # Write more unknown values (1 UInt16)
    file.write(struct.pack("<H", 18))  # From observed files

    # Write build string with version.
    build_string = f"Jul 23 2023, 19:09:15, Release, JMP, Version {version}".encode("utf-8")
    file.write(struct.pack("<i", len(build_string)))
    file.write(build_string)


def write_post_build_table_object(file: BinaryIO, n_visible: int, n_hidden: int) -> None:
    """
    Write the small JMP 17 table-object record that precedes column metadata.
    """
    record_len = _column_visibility_record_len(n_visible, n_hidden)
    object_size = record_len + 56
    file.write(struct.pack("<HIHHHII", 5, 4, 0, 1, 3, object_size, 0))


def write_column_metadata(file: BinaryIO, df: pd.DataFrame) -> int:
    """
    Write metadata about columns

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    df : pd.DataFrame
        DataFrame being written

    Returns:
    --------
    int
        File position where the Int64 column-offset table starts
    """
    # For now, all columns are visible.
    n_visible = len(df.columns)
    n_hidden = 0

    write_post_build_table_object(file, n_visible, n_hidden)

    # Write column visibility metadata section marker. The length field and
    # object ids are modeled on simple native JMP 17 data tables.
    file.write(b"\xff\xff")
    file.write(struct.pack("<QH", _column_visibility_record_len(n_visible, n_hidden), 0))
    file.write(struct.pack("<II", n_visible, n_hidden))
    file.write(struct.pack("<HHHH", 0, 0x0077, 0x0190, 0x02CF))

    # Write visible column indices (0-based)
    for i in range(n_visible):
        file.write(struct.pack("<I", i))

    # Write column widths (display width in JMP)
    for column_name in df.columns:
        # Default column width is based on name length
        width = min(max(len(column_name) * 8, 40), 300)
        file.write(struct.pack("<H", width))

    # Native JMP 17 files include a fixed 18-byte tagged record here. The
    # reader preserves these bytes as seven UInt32 values for now.
    file.write(b"\xfe\xff")
    file.write(struct.pack("<qHIIII", 18, 3, 116, 112, 169, 113))

    # Native JMP 17 files include one more tagged record before the offset
    # table. column_info() skips this marker and payload before reading ncols.
    file.write(b"\xfd\xff")
    file.write(struct.pack("<qHHI", 8, 1, 4, 0))

    # Write the number of columns again as a check
    file.write(struct.pack("<i", len(df.columns)))

    # Reserve space for column offsets
    offset_pos = file.tell()
    for _ in range(len(df.columns)):
        file.write(struct.pack("<q", 0))  # Placeholder for column offsets

    return offset_pos


def write_column_data(
    file: BinaryIO,
    column: pd.Series,
    offset: int,
    column_name: str,
    compress: bool = True,
) -> None:
    """
    Write a single column's data to the file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    offset : int
        File offset where this column should start
    column_name : str
        Name of the column
    compress : bool, default=True
        Whether to compress the data
    """
    # Seek to the column's offset
    file.seek(offset)

    # Write column name
    encoded_name = str(column_name).encode("utf-8")
    file.write(struct.pack("<h", len(encoded_name)))
    file.write(encoded_name)

    # Determine data type and write appropriate type markers
    data_type = get_column_data_type(column)

    if data_type == "float":
        write_float_column(file, column, compress)
    elif data_type == "int":
        write_int_column(file, column, compress)
    elif data_type == "string":
        write_string_column(file, column, compress)
    elif data_type == "datetime":
        write_datetime_column(file, column, compress)
    elif data_type == "date":
        write_date_column(file, column, compress)
    elif data_type == "time":
        write_time_column(file, column, compress)
    elif data_type == "duration":
        write_duration_column(file, column, compress)
    elif data_type == "rowstate":
        write_rowstate_column(file, column, compress)
    else:
        # Default to writing as string
        write_string_column(file, column, compress)


def get_column_data_type(column: pd.Series) -> str:
    """
    Determine the JMP data type for a pandas column

    Parameters:
    -----------
    column : pd.Series
        Column to analyze

    Returns:
    --------
    str
        String identifier of the column type
    """
    # Get pandas dtype
    dtype = column.dtype

    # Check for pandas extension types
    if hasattr(dtype, "name"):
        dtype_name = dtype.name

        if dtype_name == "string" or dtype_name.startswith("string"):
            return "string"

        # Check for pandas nullable integer types
        if dtype_name in ["Int8", "Int16", "Int32", "Int64"]:
            return "int"

        # Check for pandas datetime types
        if dtype_name.startswith("datetime"):
            # Try to determine if it's a date or datetime
            try:
                if all(pd.notna(t) and t.time() == time(0, 0) for t in pd.to_datetime(column.dropna())):
                    return "date"
                return "datetime"
            except (AttributeError, TypeError):
                return "datetime"

    # Check numpy/pandas basic types
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return "string"

    if np.issubdtype(np_dtype, np.datetime64):
        # Try to distinguish date from datetime
        try:
            # Check if all values have time component equal to midnight
            has_time = False
            for val in pd.to_datetime(column.dropna()):
                time_part = val.time()
                if time_part.hour != 0 or time_part.minute != 0 or time_part.second != 0:
                    has_time = True
                    break

            if has_time:
                return "datetime"
            else:
                return "date"
        except (AttributeError, TypeError):
            return "datetime"
    elif np.issubdtype(np_dtype, np.timedelta64):
        return "duration"
    elif np.issubdtype(np_dtype, np.integer):
        return "int"
    elif np.issubdtype(np_dtype, np.floating):
        return "float"
    elif dtype == "object":
        # Check first non-null value type
        non_null = column.dropna()
        if len(non_null) == 0:
            return "string"  # Default for empty series

        sample = non_null.iloc[0]

        if isinstance(sample, str):
            return "string"
        elif isinstance(sample, (datetime, np.datetime64)):
            return "datetime"
        elif isinstance(sample, date) and not isinstance(sample, datetime):
            return "date"
        elif isinstance(sample, time):
            return "time"
        elif isinstance(sample, timedelta):
            return "duration"
        elif isinstance(sample, RowState):
            return "rowstate"

    # Default to string for any other types
    return "string"


def write_float_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a float column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for float64. Compressed numeric columns in native JMP 17
    # files use the 0x63 storage markers before the gzip payload.
    if compress:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x0A, 0x00, 0x0C, 0x63, 0x63, 0x08
    else:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x01, 0x01, 0x00, 0x00, 0x00, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert to numpy array, handling missing values
    data = column.fillna(np.nan).to_numpy(dtype=np.float64)

    if compress:
        _write_compressed_column_prefix(file)

        compressed_data = _gzip_compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data directly
        file.write(data.tobytes())


def write_int_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write an integer column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Determine integer size
    dtype_name = str(column.dtype).lower()

    if "int8" in dtype_name:
        element_size = 1
        np_dtype = np.int8
        dt6 = 0x01
    elif "int16" in dtype_name:
        element_size = 2
        np_dtype = np.int16
        dt6 = 0x02
    elif "int32" in dtype_name:
        element_size = 4
        np_dtype = np.int32
        dt6 = 0x04
    else:
        # Default to int8 for other integer types if values are small enough
        min_val = column.min() if not column.empty and not column.isna().all() else 0
        max_val = column.max() if not column.empty and not column.isna().all() else 0

        if min_val >= -128 and max_val <= 127:
            element_size = 1
            np_dtype = np.int8
            dt6 = 0x01
        elif min_val >= -32768 and max_val <= 32767:
            element_size = 2
            np_dtype = np.int16
            dt6 = 0x02
        else:
            element_size = 4
            np_dtype = np.int32
            dt6 = 0x04

    # Type markers for integer. Native compressed integer columns use
    # compressed numeric storage markers and encode width in dt3/dt6.
    if compress:
        dt3_by_width = {1: 0x04, 2: 0x08, 4: 0x0C}
        dt1, dt2, dt3, dt4, dt5 = 0x0A, 0x00, dt3_by_width[element_size], 0x63, 0x63
    else:
        dt1, dt2, dt3, dt4, dt5 = 0x01, 0x01, 0x00, 0x00, 0x00
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # For missing values, use dtype's minimum value + 1 as sentinel
    sentinel = np.iinfo(np_dtype).min + 1

    # Convert to numpy array, replacing NaN with sentinel
    data = column.fillna(sentinel).to_numpy(dtype=np_dtype)

    if compress:
        _write_compressed_column_prefix(file)

        compressed_data = _gzip_compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * element_size))  # uncompressed size

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data directly
        file.write(data.tobytes())


def write_string_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a string column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Replace NaN with empty string
    column = column.fillna("")

    # Get string lengths
    encoded_strings = [str(s).encode("utf-8") for s in column]
    string_lengths = [len(s) for s in encoded_strings]
    max_length = max(string_lengths, default=0)
    min_length = min(string_lengths, default=0)

    if compress:
        # Native compressed character columns are stored as variable-width
        # payloads, even when every string currently has the same width.
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x09, 0x02, 0x00, 0x00, 0x00, 0x00
        file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))
        _write_compressed_column_prefix(file, property_count=4)

        use_int16 = max_length >= 128

        lengths = bytearray()
        string_data = bytearray()

        for encoded in encoded_strings:
            string_data.extend(encoded)

            if use_int16:
                lengths.extend(struct.pack("<h", len(encoded)))
            else:
                lengths.extend(struct.pack("<b", len(encoded)))

        header = bytearray([0] * 13)
        header[8] = 2 if use_int16 else 1  # Width bytes
        struct.pack_into(
            "<q", header, 0, len(header) + len(lengths) + len(string_data) - 8
        )
        struct.pack_into("<i", header, 9, max_length)

        all_data = header + lengths + string_data
        compressed_data = _gzip_compress(all_data)

        file.write(GZIP_SECTION_START)
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(all_data)))  # uncompressed size
        file.write(compressed_data)
        return

    # Determine if we should use fixed width or variable width
    use_fixed_width = max_length == min_length and 0 < max_length <= 255

    if use_fixed_width:
        # For fixed width strings
        dt1 = 0x02
        dt2, dt3, dt4 = 0x01, 0x00, 0x00
        dt5 = max_length
        dt6 = 0x00

        file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

        # Create a byte array of fixed width strings
        string_data = bytearray()
        for encoded in encoded_strings:
            # Encode the string as UTF-8 and pad/truncate to fixed width
            padded = encoded[:dt5].ljust(dt5, b"\0")
            string_data.extend(padded)

        # Write raw string data
        file.write(string_data)
    else:
        # For variable width strings
        dt1 = 0x09
        dt2, dt3, dt4, dt5, dt6 = 0x02, 0x00, 0x00, 0x00, 0x00

        file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

        # Prepare variable width string data

        # Determine width needed for length values
        use_int16 = max_length >= 128

        # Prepare lengths and string data
        lengths = bytearray()
        string_data = bytearray()

        for encoded in encoded_strings:
            string_data.extend(encoded)

            if use_int16:
                lengths.extend(struct.pack("<h", len(encoded)))
            else:
                lengths.extend(struct.pack("<b", len(encoded)))

        # Prepare header
        header = bytearray([0] * 13)
        header[8] = 2 if use_int16 else 1  # Width bytes
        struct.pack_into(
            "<q", header, 0, len(header) + len(lengths) + len(string_data) - 8
        )
        struct.pack_into("<i", header, 9, max_length)

        # Combine all data
        all_data = header + lengths + string_data

        compressed_data = _gzip_compress(all_data)

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(all_data)))  # uncompressed size

        # Write the compressed data
        file.write(compressed_data)


def write_datetime_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a datetime column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for datetime
    if compress:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x0A, 0x00, 0x16, 0x7E, 0x7E, 0x08
    else:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x01, 0x01, 0x00, 0x69, 0x69, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert to seconds since JMP epoch
    def to_jmp_time(dt):
        if pd.isna(dt):
            return np.nan

        if isinstance(dt, (np.datetime64, pd.Timestamp)):
            dt = pd.Timestamp(dt).to_pydatetime()

        return (dt - JMP_STARTDATE).total_seconds()

    epoch_seconds = column.apply(to_jmp_time)

    # Convert to numpy array
    data = epoch_seconds.to_numpy(dtype=np.float64)

    if compress:
        _write_compressed_column_prefix(file)

        compressed_data = _gzip_compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_date_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a date column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for date
    if compress:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x0A, 0x00, 0x0C, 0x7F, 0x7F, 0x08
    else:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x01, 0x01, 0x00, 0x65, 0x65, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert to seconds since JMP epoch (with time part set to 00:00:00)
    def to_jmp_date(d):
        if pd.isna(d):
            return np.nan

        if isinstance(d, (np.datetime64, pd.Timestamp)):
            d = pd.Timestamp(d).to_pydatetime().date()
        elif isinstance(d, datetime):
            d = d.date()

        return (datetime.combine(d, time.min) - JMP_STARTDATE).total_seconds()

    epoch_seconds = column.apply(to_jmp_date)

    # Convert to numpy array
    data = epoch_seconds.to_numpy(dtype=np.float64)

    if compress:
        _write_compressed_column_prefix(file)

        compressed_data = _gzip_compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_time_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a time column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for time
    if compress:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x0A, 0x00, 0x0C, 0x82, 0x82, 0x08
    else:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x01, 0x01, 0x00, 0x82, 0x82, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert time to seconds since midnight
    def to_seconds(t):
        if pd.isna(t):
            return np.nan

        if isinstance(t, (datetime, pd.Timestamp)):
            t = t.time()

        return t.hour * 3600 + t.minute * 60 + t.second

    seconds = column.apply(to_seconds)

    # Convert to numpy array
    data = seconds.to_numpy(dtype=np.float64)

    if compress:
        _write_compressed_column_prefix(file)

        compressed_data = _gzip_compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_duration_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a duration column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for duration
    if compress:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x0A, 0x00, 0x0C, 0x85, 0x85, 0x08
    else:
        dt1, dt2, dt3, dt4, dt5, dt6 = 0x01, 0x01, 0x00, 0x6C, 0x6C, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert duration to seconds
    def to_seconds(d):
        if pd.isna(d):
            return np.nan

        if isinstance(d, (timedelta, pd.Timedelta)):
            return d.total_seconds()

        return float(d)  # Attempt to convert other types

    seconds = column.apply(to_seconds)

    # Convert to numpy array
    data = seconds.to_numpy(dtype=np.float64)

    if compress:
        _write_compressed_column_prefix(file)

        compressed_data = _gzip_compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_rowstate_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a row state column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write (containing RowState objects)
    compress : bool
        Whether to compress the data
    """
    # Type markers for row state
    dt1 = 0x09  # Always use compressed format for row states
    dt2, dt3, dt4 = 0x03, 0x00, 0x00
    dt5 = 8  # Width of each row state entry
    dt6 = 0x00
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Prepare row state data
    row_data = bytearray()

    for rs in column:
        if pd.isna(rs):
            # Default row state for missing values
            marker_idx = 0  # First marker
            color_idx = 0  # First color (black)
            data_entry = bytearray([0, color_idx, 0, 0, 0, 0, 0, marker_idx])
            row_data.extend(data_entry)
            continue

        # Get marker index
        try:
            marker_idx = ROWSTATE_MARKERS.index(rs.marker)

            # Write as UInt16 in little-endian format (lower byte first)
            marker_bytes = marker_idx.to_bytes(2, byteorder="little")
        except ValueError:
            # If marker not found in predefined list, use character code
            marker_bytes = ord(rs.marker).to_bytes(2, byteorder="little")

        # Get RGB color components
        r, g, b = rs.color

        # Check if the color is in the predefined list
        hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}".upper()
        try:
            color_idx = ROWSTATE_COLORS.index(hex_color)
            custom_color = False
        except ValueError:
            # Use custom color encoding
            color_idx = 0
            custom_color = True

        # Build row state entry
        if custom_color:
            # Custom color format
            data_entry = bytearray(
                [
                    0,  # Unknown
                    int(b * 255),  # Blue
                    int(g * 255),  # Green
                    int(r * 255),  # Red
                    0xFF,  # Flag for custom color
                    0,  # Unknown
                    marker_bytes[0],  # Marker (low byte)
                    marker_bytes[1],  # Marker (high byte)
                ]
            )
        else:
            # Predefined color format
            data_entry = bytearray(
                [
                    0,  # Unknown
                    color_idx,  # Color index
                    0,  # Unknown
                    0,  # Unknown
                    0,  # Flag for predefined color
                    0,  # Unknown
                    marker_bytes[0],  # Marker (low byte)
                    marker_bytes[1],  # Marker (high byte)
                ]
            )

        row_data.extend(data_entry)

    # Compress row state data
    compressed_data = _gzip_compress(row_data)
    _write_compressed_column_prefix(file)

    # Write gzip section marker
    file.write(GZIP_SECTION_START)

    # Write compressed and uncompressed sizes
    file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
    file.write(struct.pack("<Q", len(row_data)))  # uncompressed size

    # Write the compressed data
    file.write(compressed_data)


def finalize_file(file: BinaryIO) -> None:
    """
    Perform any final operations on the file before closing

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    """
    # This function can be used for any final adjustments to the file
    # Ensure file is properly aligned on 8-byte boundary
    current_pos = file.tell()
    padding = 8 - (current_pos % 8) if current_pos % 8 != 0 else 0

    if padding > 0:
        file.write(bytearray([0] * padding))

    # We may need to update some file sizes or checksums here
    # For now, no additional finalization is needed
