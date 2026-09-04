"""
jmpio - A package for reading and writing SAS JMP files

This package provides functionality to read and write binary JMP files from
the SAS JMP statistical software.
"""

from .jsl import (
    ColumnProperties,
    JSLAutomationError,
    JSLResult,
    apply_column_properties_with_jsl,
    dump_column_properties_with_jsl,
    find_jmp_executable,
    validate_jmp_with_jsl,
    write_jmp_with_jsl,
)
from .reader import read_jmp, scan_directory
from .writer import write_jmp

__all__ = [
    "ColumnProperties",
    "JSLAutomationError",
    "JSLResult",
    "apply_column_properties_with_jsl",
    "dump_column_properties_with_jsl",
    "find_jmp_executable",
    "read_jmp",
    "scan_directory",
    "validate_jmp_with_jsl",
    "write_jmp",
    "write_jmp_with_jsl",
]
__version__ = "0.3.0"
