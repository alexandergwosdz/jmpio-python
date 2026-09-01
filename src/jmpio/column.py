"""
Functions for reading column data from JMP files
"""

import gzip
import io
import struct
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from .constants import (
    GZIP_SECTION_START,
    JMP_STARTDATE,
    ROWSTATE_COLORS,
    ROWSTATE_MARKERS,
)
from .types import JMPInfo, RowState
from .utils import bit_cat, hex_to_rgb, read_string, sentinel_to_missing, to_datetime


def read_column_data(file: BinaryIO, info: JMPInfo, column_idx: int) -> Any:
    if not (0 <= column_idx < info.ncols):
        raise ValueError(f"Column index {column_idx} is out of bounds (0-{info.ncols - 1})")

    file.seek(info.column.offsets[column_idx])
    column_name = read_string(file, 2)

    dt_bytes = file.read(6)
    if len(dt_bytes) < 6:
        raise EOFError(f"Could not read 6 dt_bytes for column {column_name}")
    dt = struct.unpack("BBBBBB", dt_bytes)
    dt1, dt2, dt3, dt4, dt5, dt6 = dt

    start_pos_after_dt = file.tell()

    is_compressed = dt1 in [0x09, 0x0A]
    data_payload_source: io.BytesIO | None = None
    col_end = -1

    if not is_compressed:
        if column_idx == info.ncols - 1:
            current_file_pos_for_col_end = file.tell()
            file.seek(0, 2)
            col_end = file.tell()
            file.seek(current_file_pos_for_col_end)
        else:
            col_end = info.column.offsets[column_idx + 1]

    if is_compressed:
        original_file_pos = file.tell()
        header_search_limit = 2048
        temp_header = file.read(header_search_limit)
        gzip_section_offset = temp_header.find(GZIP_SECTION_START)

        if gzip_section_offset == -1:
            raise EOFError(f"GZIP_SECTION_START not found within {header_search_limit} bytes for compressed column {column_name}")

        file.seek(original_file_pos + gzip_section_offset)
        gs_check = file.read(len(GZIP_SECTION_START))
        if gs_check != GZIP_SECTION_START:
            raise ValueError("GZIP_SECTION_START check failed.")

        gzip_len = struct.unpack("<Q", file.read(8))[0]
        gunzip_len = struct.unpack("<Q", file.read(8))[0]  # noqa: F841 # Unused, but must be read.
        compressed_data_bytes = file.read(gzip_len)

        try:
            decompressed_bytes = gzip.decompress(compressed_data_bytes)
            data_payload_source = io.BytesIO(decompressed_bytes)
        except Exception as e:
            raise ValueError(f"Failed to decompress data for column '{column_name}': {str(e)}")

    try:
        if dt1 in [0x01, 0x0A]:  # Numeric types
            if not is_compressed:
                file.seek(start_pos_after_dt)
                uncompressed_data_block_bytes = file.read(col_end - start_pos_after_dt)
                data_payload_source = io.BytesIO(uncompressed_data_block_bytes)

            if data_payload_source is None:
                raise ValueError("data_payload_source not initialized for numeric type")

            dtype_map = {0x01: np.int8, 0x02: np.int16, 0x04: np.int32}
            selected_dtype = dtype_map.get(dt6, np.float64)
            item_size = np.dtype(selected_dtype).itemsize
            data_size = item_size * info.nrows

            data_payload_source.seek(-data_size, 2)
            raw_data = data_payload_source.read(data_size)
            if len(raw_data) < data_size:
                raise EOFError(f"Not enough data for numeric column {column_name}. Expected {data_size}, got {len(raw_data)}")
            data_array = np.frombuffer(raw_data, dtype=selected_dtype)
            series_data = sentinel_to_missing(np.copy(data_array))

            if (dt4 == dt5 and dt4 in [0x00, 0x03, 0x42, 0x43, 0x44, 0x59, 0x60, 0x63]) or dt5 in [0x5E, 0x63]:
                return series_data
            if dt4 == dt5 and dt4 in [0x5F]:
                return series_data  # Currency
            if dt4 == dt5 and dt4 in [0x54, 0x55, 0x56]:
                return series_data  # Longitude
            if dt4 == dt5 and dt4 in [0x51, 0x52, 0x53]:
                return series_data  # Latitude

            datetime_array = to_datetime(data_array.astype(np.float64))

            date_dt4_idbs = [ 0x65, 0x66, 0x67, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x75, 0x76, 0x7A, 0x7F, 0x88, 0x8B,]

            if (
                dt4 == dt5
                and dt4
                in date_dt4_idbs
            ) or any(
                np.array_equal([dt4, dt5], p)
                for p in [
                    [0x67, 0x65],
                    [0x6F, 0x65],
                    [0x72, 0x65],
                    [0x72, 0x6F],
                    [0x72, 0x7F],
                    [0x72, 0x80],
                    [0x7F, 0x72],
                    [0x88, 0x65],
                    [0x88, 0x7A],
                ]
            ):
                return pd.Series(datetime_array.astype("datetime64[D]"))  # Date


            if (
                (
                    dt5 in [0x69, 0x6A, 0x73, 0x74, 0x77, 0x78, 0x7E, 0x81]
                    and dt4
                    in [
                        0x69,
                        0x6A,
                        0x6C,
                        0x6D,
                        0x73,
                        0x74,
                        0x77,
                        0x78,
                        0x79,
                        0x7B,
                        0x7C,
                        0x7D,
                        0x7E,
                        0x80,
                        0x81,
                        0x82,
                        0x86,
                        0x87,
                        0x89,
                        0x8A,
                    ]
                )
                or (dt4 == dt5 and dt4 in [0x79, 0x7D])
                or any(np.array_equal([dt4, dt5], p) for p in [[0x77, 0x80], [0x77, 0x7F], [0x89, 0x65]])
            ):
                return pd.Series(datetime_array)  # DateTime
            if dt4 == dt5 and dt4 in [0x82]:  # Time
                return pd.Series(pd.to_datetime(datetime_array).time)
            if (dt4 == dt5 and dt4 in [0x0C, 0x6B, 0x6C, 0x6D, 0x83, 0x84, 0x85]) or any(
                np.array_equal([dt4, dt5], p) for p in [[0x84, 0x79]]
            ):  # Duration
                epoch_start = np.datetime64(JMP_STARTDATE)
                valid_dates = pd.to_datetime(datetime_array, errors="coerce")
                delta = valid_dates - epoch_start
                return pd.Series(delta)

        elif dt1 in [0xFF, 0xFE, 0xFC]:  # Alternative byte integer
            if not is_compressed:
                file.seek(start_pos_after_dt)
                uncompressed_data_block_bytes = file.read(col_end - start_pos_after_dt)
                data_payload_source = io.BytesIO(uncompressed_data_block_bytes)

            if data_payload_source is None:
                raise ValueError("data_payload_source not initialized for alt byte int type")

            dtype_map = {0x01: np.int8, 0x02: np.int16, 0x04: np.int32}
            selected_dtype = dtype_map.get(dt5, np.float64)
            item_size = np.dtype(selected_dtype).itemsize
            data_size = item_size * info.nrows

            data_payload_source.seek(-data_size, 2)
            raw_data = data_payload_source.read(data_size)
            if len(raw_data) < data_size:
                raise EOFError(f"Not enough data for alt int column {column_name}. Expected {data_size}, got {len(raw_data)}")
            data_array = np.frombuffer(raw_data, dtype=selected_dtype)
            return sentinel_to_missing(np.copy(data_array))

        elif dt1 == 0x09 and dt2 == 0x03:  # Row states
            if not is_compressed or data_payload_source is None:
                raise ValueError("Row states expect compressed data_payload_source")

            width = dt5
            row_states = []
            data_payload_source.seek(0)

            for _ in range(info.nrows):
                row_data_bytes = data_payload_source.read(width)
                if len(row_data_bytes) < width:
                    raise EOFError(f"Not enough data for row state in {column_name}")

                marker_idx = bit_cat(row_data_bytes[6], row_data_bytes[5])
                marker = ROWSTATE_MARKERS[marker_idx] if marker_idx < len(ROWSTATE_MARKERS) else chr(marker_idx)

                r, g, b = 0.0, 0.0, 0.0
                if row_data_bytes[3] == 0xFF:
                    r = row_data_bytes[2] / 255.0
                    g = row_data_bytes[1] / 255.0
                    b = row_data_bytes[0] / 255.0
                else:
                    color_idx = row_data_bytes[0]
                    if 0 <= color_idx < len(ROWSTATE_COLORS):
                        hex_color = ROWSTATE_COLORS[color_idx]
                        r, g, b = hex_to_rgb(hex_color)
                row_states.append(RowState(marker=marker, color=(r, g, b)))
            return pd.Series(row_states)

        # Character data
        elif dt1 in [0x02, 0x09] and dt2 in [0x01, 0x02]:
            # Constant width
            if ([dt3, dt4] == [0x00, 0x00] and dt5 > 0) or (0x01 <= dt3 <= 0x07 and dt4 == 0x00):
                if not is_compressed:
                    file.seek(start_pos_after_dt)
                    uncompressed_data_block_bytes = file.read(col_end - start_pos_after_dt)
                    data_payload_source = io.BytesIO(uncompressed_data_block_bytes)

                if data_payload_source is None:
                    raise ValueError("data_payload_source not initialized for const char type")

                width = dt5
                strings = []
                data_size = width * info.nrows
                data_payload_source.seek(-data_size, 2)
                all_string_data = data_payload_source.read(data_size)
                if len(all_string_data) < data_size:
                    raise EOFError(f"Not enough data for const char column {column_name}")

                for i in range(info.nrows):
                    start = i * width
                    s_bytes = all_string_data[start : start + width]
                    s = s_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                    strings.append(s)
                return pd.Series(strings)

            # Variable width
            elif [dt3, dt4, dt5] == [0x00, 0x00, 0x00]:
                strings = []
                if is_compressed:
                    if data_payload_source is None:
                        raise ValueError("data_payload_source not initialized for compressed var char")

                    # Get full decompressed payload bytes
                    payload = data_payload_source.getvalue()

                    # Pooled data check: first 8 bytes (little-endian Int64) equal total payload length
                    if len(payload) >= 8 and struct.unpack("<q", payload[:8])[0] == len(payload):
                        buf = io.BytesIO(payload)
                        _reclen = struct.unpack("<q", buf.read(8))[0]
                        _foo = buf.read(9)
                        _foo2 = struct.unpack("<q", buf.read(8))[0]

                        # Indices to pool
                        wb = buf.read(1)[0]
                        if wb == 1:
                            idx_dtype = np.uint8
                            idx_itemsize = 1
                        elif wb == 2:
                            idx_dtype = np.dtype("<u2")
                            idx_itemsize = 2
                        elif wb == 4:
                            idx_dtype = np.dtype("<u4")
                            idx_itemsize = 4
                        else:
                            raise ValueError(f"Unknown index width byte {wb} for pooled var char in {column_name}")

                        idx_raw = buf.read(idx_itemsize * info.nrows)
                        idx = np.frombuffer(idx_raw, dtype=idx_dtype)

                        # Pool definitions
                        npools = int(idx.max(initial=0)) if hasattr(idx, "max") else 0
                        wb2 = buf.read(1)[0]
                        if wb2 == 1:
                            pool_len_fmt = "b"
                            pool_len_size = 1
                        elif wb2 == 2:
                            pool_len_fmt = "<h"
                            pool_len_size = 2
                        else:
                            raise ValueError(f"Unknown pool length width byte {wb2} for pooled var char in {column_name}")

                        pool: list[str] = []
                        for _ in range(npools):
                            n = struct.unpack(pool_len_fmt, buf.read(pool_len_size))[0]
                            if n < 0:
                                n = 0
                            s_bytes = buf.read(n)
                            pool.append(s_bytes.decode("utf-8", errors="replace"))

                        for idx_val in idx.tolist():
                            strings.append("" if idx_val == 0 else pool[int(idx_val) - 1])
                        return pd.Series(strings)

                    # Non-pooled compressed variable-width strings
                    data_payload_source.seek(8)
                    width_bytes_val = data_payload_source.read(1)[0]
                    lengths_offset_in_payload = 13
                    data_payload_source.seek(lengths_offset_in_payload)

                    if width_bytes_val == 1:
                        len_dtype, bytes_for_lengths = np.int8, info.nrows
                    elif width_bytes_val == 2:
                        len_dtype, bytes_for_lengths = np.int16, 2 * info.nrows
                    elif width_bytes_val == 4:
                        len_dtype, bytes_for_lengths = np.int32, 4 * info.nrows
                    else:
                        raise ValueError(f"Unknown width_bytes_val {width_bytes_val} for compressed var char")

                    lengths_raw = data_payload_source.read(bytes_for_lengths)
                    lengths = np.frombuffer(lengths_raw, dtype=len_dtype)
                    string_data_block = data_payload_source.read()
                    current_offset_in_strings = 0
                    for length in lengths:
                        length_int = int(length)
                        s_bytes = string_data_block[current_offset_in_strings : current_offset_in_strings + length_int]
                        strings.append(s_bytes.decode("utf-8", errors="replace"))
                        current_offset_in_strings += length_int
                    return pd.Series(strings)
                else:
                    # Uncompressed variable width
                    file.seek(start_pos_after_dt)
                    _ = file.read(6)
                    n1 = struct.unpack("<q", file.read(8))[0]
                    _ = file.read(n1)
                    _ = file.read(2)
                    n2 = struct.unpack("<I", file.read(4))[0]
                    _ = file.read(n2 + 8)
                    width_bytes_val = struct.unpack("B", file.read(1))[0]
                    _ = file.read(4)  # Skip max_width

                    if width_bytes_val == 1:
                        len_dtype, itemsize_for_len = np.int8, 1
                    elif width_bytes_val == 2:
                        len_dtype, itemsize_for_len = np.int16, 2
                    elif width_bytes_val == 4:
                        len_dtype, itemsize_for_len = np.int32, 4
                    else:
                        raise ValueError(f"Unknown width_bytes_val {width_bytes_val} for uncompressed var char")

                    lengths_raw = file.read(itemsize_for_len * info.nrows)
                    lengths = np.frombuffer(lengths_raw, dtype=len_dtype)
                    sum_lengths = int(np.sum(lengths))
                    if sum_lengths < 0:
                        sum_lengths = 0

                    file.seek(col_end - sum_lengths)
                    all_string_data_bytes = file.read(sum_lengths)
                    current_offset = 0
                    for length in lengths:
                        length_int = int(length)
                        s_bytes = all_string_data_bytes[current_offset : current_offset + length_int]
                        strings.append(s_bytes.decode("utf-8", errors="replace"))
                        current_offset += length_int
                    return pd.Series(strings)

        elif dt1 == 0x03 and dt2 == 0x03:
            if not is_compressed:
                file.seek(start_pos_after_dt)
                uncompressed_data_block_bytes = file.read(col_end - start_pos_after_dt)
                data_payload_source = io.BytesIO(uncompressed_data_block_bytes)

            if data_payload_source is None:
                raise ValueError("data_payload_source not initialized for Int64 row state type")

            width = dt5
            data_size = width * info.nrows
            data_payload_source.seek(-data_size, 2)
            raw_data = data_payload_source.read(data_size)
            if len(raw_data) < data_size:
                raise EOFError(f"Not enough data for Int64 row state column {column_name}")
            return pd.Series(np.frombuffer(raw_data, dtype=np.dtype("<i8")))

    except struct.error as e:
        print(
            f"Struct unpacking error in column '{column_name}' (idx={column_idx}): {str(e)}. File position: {file.tell() if not file.closed else 'closed'}"
        )
    except EOFError as e:
        print(f"EOFError in column '{column_name}' (idx={column_idx}): {str(e)}")
    except Exception as e:
        print(
            f"Generic error parsing column '{column_name}' (idx={column_idx}), dt=({dt1:02x},{dt2:02x},{dt3:02x},{dt4:02x},{dt5:02x},{dt6:02x}): {str(e)}"
        )

    print(
        f"Unknown or unhandled data type combination: dt=({dt1:02x},{dt2:02x},{dt3:02x},{dt4:02x},{dt5:02x},{dt6:02x}) "
        f"for column {column_name} (idx={column_idx})"
    )
    return pd.Series([np.nan] * info.nrows)
