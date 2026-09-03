"""
jmpio - A package for reading and writing SAS JMP files

This package provides functionality to read and write binary JMP files from
the SAS JMP statistical software.
"""

from .jsl import (
    JSLAutomationError,
    JSLResult,
    find_jmp_executable,
    validate_jmp_with_jsl,
    write_jmp_with_jsl,
)
from .reader import read_jmp, scan_directory
from .writer import write_jmp

__all__ = [
    "JSLAutomationError",
    "JSLResult",
    "find_jmp_executable",
    "read_jmp",
    "scan_directory",
    "validate_jmp_with_jsl",
    "write_jmp",
    "write_jmp_with_jsl",
]
__version__ = "0.2.0"
