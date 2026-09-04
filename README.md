# jmpio

A Python package for reading and writing SAS JMP files.

## Description

`jmpio` is a mostly-faithful Python port of the Julia package [JMPReader.jl](https://github.com/jaakkor2/JMPReader.jl) by [@jaakkor2](https://github.com/jaakkor2). It aims to be able to read and write binary JMP files from SAS JMP statistical software. There's no way this package would exist without @jaakor2's efforts.

> [!WARNING]
> `jmpio` write support is still experimental.
> `jmpio` is not particularly efficient and uses pandas DataFrames internally.

## Features

- Read JMP files into pandas DataFrames
- Write pandas DataFrames to JMP files (experimental)
  - Optional JSL workflow using a local JMP installation
  - Embed Scripts in those files for plotting data.
- Support for all JMP data types:
  - Numeric (Float64, Int8, Int16, Int32)
  - Character (fixed width and variable width strings)
  - Date/Time (Date, DateTime, Time, Duration)
  - Geographic (Latitude/Longitude)
  - Row states (markers and colors)
  - Currency
- Support for compressed and uncompressed JMP files
- Column selection/filtering when reading
- Strong typing throughout the codebase

## Installation

```bash
pip install jmpio
```

## Basic Usage

### Reading a JMP file

```python
import jmpio
import pandas as pd

# Read a JMP file into a pandas DataFrame
df = jmpio.read_jmp("path/to/file.jmp")

# Select specific columns
df = jmpio.read_jmp("path/to/file.jmp", select=["Column1", "Column2"])

# Drop specific columns
df = jmpio.read_jmp("path/to/file.jmp", drop=["Column1", "Column2"])

# Find all JMP files in a directory
file_info = jmpio.scan_directory("path/to/directory")
```

### Writing a JMP file

```python
import jmpio
import pandas as pd
from datetime import datetime, date

# Create a DataFrame
df = pd.DataFrame({
    'integers': [1, 2, 3, 4],
    'floats': [1.1, 2.2, 3.3, 4.4],
    'strings': ['a', 'bb', 'ccc', 'dddd'],
    'dates': [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)]
})

# Write to a JMP file with compression using the native Python writer (default)
jmpio.write_jmp(df, "output.jmp")

# Write to a JMP file without compression
jmpio.write_jmp(df, "output_uncompressed.jmp", compress=False)

# Use a local JMP installation through generated JSL scripts
jmpio.write_jmp(df, "output_native.jmp", engine="jsl")

# Write with the Python writer, validate with JMP when available, and fall back
# to JSL if JMP cannot open the Python-written file
jmpio.write_jmp(df, "output_auto.jmp", engine="auto")
```

The JSL workflow is optional and requires SAS JMP to be installed locally. By
default, `jmpio` searches common Windows install paths and `JMPIO_JMP_EXE`; you
can also pass `jmp_executable="C:/Program Files/SAS/JMP/17/jmp.exe"`.

### Applying JMP column properties

When JMP is installed locally, `jmpio` can use JSL automation to apply or dump
column properties that are not written by the native Python binary writer:

```python
import jmpio

jmpio.apply_column_properties_with_jsl(
    "output.jmp",
    [
        {
            "column": "Protein Concentration (mg/mL)",
            "lsl": 90,
            "usl": 110,
            "show_limits": 0,
        },
        {
            "column": "Aggregate (%)",
            "control_limits": "{XBar(Avg(0.4), LCL(0.35), UCL(0.55))}",
        },
        {"column": "Total Purity (%)", "sigma": "0.007"},
        {
            "column": "PC^2",
            "formula": ':"Protein Concentration (mg/mL)"n * :"Protein Concentration (mg/mL)"n',
        },
    ],
)

properties = jmpio.dump_column_properties_with_jsl("output.jmp")
```

Supported property fields are `lsl`, `usl`, `target`, `show_limits`,
`control_limits`, `sigma`, and `formula`.

## Supported Data Types

### Reading

The following data types are supported when reading JMP files:

| JMP Data Type | Python Data Type |
|---------------|------------------|
| Numeric       | float64, int8, int16, int32 |
| Character     | str (fixed or variable width) |
| Date          | datetime64[D] / datetime.date |
| Time          | datetime64[s] / datetime.time |
| DateTime      | datetime64[s] / datetime.datetime |
| Duration      | timedelta64[ms] / datetime.timedelta |
| Row State     | jmpio.RowState |
| Geographic    | float64 |
| Currency      | float64 |

### Writing

The following data types are supported when writing to JMP files:

| Python Data Type | JMP Data Type |
|------------------|---------------|
| int, Int8, Int16, Int32 | Integer |
| float, Float64 | Numeric |
| str | Character |
| datetime64, datetime.datetime | DateTime |
| datetime64[D], datetime.date | Date |
| datetime.time | Time |
| timedelta64, datetime.timedelta | Duration |
| jmpio.RowState | Row State |

## Requirements

- Python 3.10+
- NumPy
- pandas

## License

MIT

## Acknowledgments

This package is a Python port of the Julia package [JMPReader.jl](https://github.com/jaakkor2/JMPReader.jl).
JMP is a registered trademark of SAS Institute Inc. This project is not affiliated with, sponsored by, or endorsed by SAS Institute Inc.


