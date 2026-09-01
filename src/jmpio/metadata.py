"""
Functions for reading JMP file metadata
"""

import re
import struct
from typing import BinaryIO

import numpy as np
import pandas as pd

from .types import Column, JMPInfo
from .utils import find_sequence_in_file, read_reals, read_string, to_datetime


def read_metadata(file: BinaryIO) -> JMPInfo:
    """
    Read metadata from a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle to a JMP file

    Returns:
    --------
    JMPInfo
        Information about the JMP file
    """
    # Position the file cursor like the Julia implementation:
    # seq = [0x07, 0x00, 0x08, 0x00, 0x00, 0x00]; readuntil(io, seq); seek(io, position(io) - 38)
    # We'll scan for this sequence from the beginning of the file and then
    # back up 38 bytes from the end of the match.
    seq = bytes([0x07, 0x00, 0x08, 0x00, 0x00, 0x00])
    idx = find_sequence_in_file(file, seq, start=0, stop=-1, chunk_size=4096)
    if idx is None:
        raise ValueError("Could not find JMP metadata sequence in file")
    else:
        # Position after sequence then move back 38 bytes
        file.seek(idx + len(seq) - 38)
    # Read number of rows and columns
    nrows = struct.unpack("<q", file.read(8))[0]  # Int64, little-endian
    ncols = struct.unpack("<i", file.read(4))[0]  # Int32, little-endian
    # Read some unknown values (little-endian Int16)
    foo1 = read_reals(file, np.dtype("<i2"), 5)
    # Read character set
    charset = read_string(file, 4).rstrip("\0")

    # More unknown values (little-endian UInt16)
    foo2 = read_reals(file, np.dtype("<u2"), 3)

    # Read save time
    save_time_float = struct.unpack("<d", file.read(8))[0]  # Float64, little-endian
    savetime_arr = to_datetime(np.array([save_time_float]))
    savetime = pd.Timestamp(savetime_arr[0]).to_pydatetime()

    # More unknown values (single little-endian UInt16)
    foo3 = read_reals(file, np.dtype("<u2"), 1)[0]

    # Build string and version
    buildstring = read_string(file, 4)
    match = re.search(r"Version (.*?)$", buildstring)
    if not match:
        raise ValueError("Could not determine JMP version")
    version = match.group(1)

    # Find offsets to column data
    n_visible, n_hidden = seek_to_column_data_offsets(file, ncols)

    # Read visible and hidden column indices
    idx_visible = read_reals(file, np.dtype("<u4"), n_visible)
    idx_hidden = read_reals(file, np.dtype("<u4"), n_hidden)

    # Read column display widths
    colwidths = read_reals(file, np.dtype("<u2"), ncols)

    # Read and preserve seven unknown 4-byte integers (UInt32)
    unknown_u32_7 = read_reals(file, np.dtype("<u4"), 7)

    # Read column names and offsets
    colnames, coloffsets = column_info(file, ncols)

    return JMPInfo(
        version=version,
        buildstring=buildstring,
        savetime=savetime,
        nrows=nrows,
        ncols=ncols,
        column=Column(names=colnames, widths=colwidths.tolist(), offsets=coloffsets),
        foo1=foo1,
        charset=charset,
        foo2=foo2,
        foo3=foo3,
        n_visible=int(n_visible),
        n_hidden=int(n_hidden),
        idx_visible=idx_visible,
        idx_hidden=idx_hidden,
        unknown_u32_7=unknown_u32_7,
    )


def column_info(file: BinaryIO, ncols: int) -> tuple[list[str], list[int]]:
    """
    Extract column names and data offsets from a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle to a JMP file
    ncols : int
        Number of columns

    Returns:
    --------
    tuple[list[str], list[int]]
        A tuple containing column names and column offsets
    """
    while True:
        twobytes = file.read(2)
        if len(twobytes) < 2:
            raise EOFError("Unexpected end of file when searching for column information")

        # Check for special marker bytes
        if twobytes in [b"\xfc\xff", b"\xfd\xff", b"\xfe\xff", b"\xff\xff"]:
            n = struct.unpack("<q", file.read(8))[0]  # Int64, little-endian
            _ = file.read(n)  # Skip this data
        else:
            # Go back two bytes
            file.seek(file.tell() - 2)
            break

    # Read number of columns again as a check
    ncols2 = struct.unpack("<i", file.read(4))[0]  # Int32, little-endian
    if ncols != ncols2:
        raise ValueError(f"Number of columns mismatch: {ncols} vs {ncols2}")

    # Read column offsets
    coloffsets = read_reals(file, np.int64, ncols).tolist()

    # Read column names
    colnames = []
    for offset in coloffsets:
        file.seek(offset)
        colnames.append(read_string(file, 2))

    return colnames, coloffsets


def seek_to_column_data_offsets(file: BinaryIO, ncols: int) -> tuple[int, int]:
    """
    Find the location of column data offsets in the file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle to a JMP file
    ncols : int
        Number of columns

    Returns:
    --------
    tuple[int, int]
        Number of visible columns and number of hidden columns
    """
    # Save current position
    orig_pos = file.tell()

    # Start from beginning
    file.seek(2)  # Skip first 2 bytes

    while True:
        # Read in chunks for efficiency
        chunk_size = 4096
        offset = file.tell()
        chunk = file.read(chunk_size)

        if not chunk:
            # Reset position and raise error if we reach EOF
            file.seek(orig_pos)
            raise ValueError("Could not find column offset data")

        # Look for 0xFF 0xFF sequence
        for i in range(len(chunk) - 1):
            if chunk[i] == 0xFF and chunk[i + 1] == 0xFF:
                # Found potential match, seek to this position
                file.seek(offset + i)

                # Skip any additional 0xFF bytes
                while file.read(1) == b"\xff":
                    pass

                # Go back one byte
                file.seek(file.tell() - 1)

                # Skip 10 bytes
                file.seek(file.tell() + 10)

                # Read number of visible and hidden columns
                n_visible = struct.unpack("<I", file.read(4))[0]  # UInt32, little-endian
                n_hidden = struct.unpack("<I", file.read(4))[0]  # UInt32, little-endian

                # Skip 8 more bytes
                file.seek(file.tell() + 8)

                # Check if we found the right location
                if n_visible + n_hidden == ncols:
                    return n_visible, n_hidden

                # If not, go back and continue searching
                file.seek(offset + i + 2)

        # Move back a bit to avoid missing a match at chunk boundary
        file.seek(offset + len(chunk) - 1)

    # Reset position before returning
    file.seek(orig_pos)
    raise ValueError("Could not find column offset data")
