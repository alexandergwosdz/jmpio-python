"""
Optional JMP automation helpers using generated JSL scripts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_JMP_EXECUTABLES = (
    r"C:\Program Files\SAS\JMP\19\jmp.exe",
    r"C:\Program Files\SAS\JMP\18\jmp.exe",
    r"C:\Program Files\SAS\JMP\17\jmp.exe",
    r"C:\Program Files\SAS\JMP\16\jmp.exe",
)


@dataclass(frozen=True)
class JSLResult:
    """Result from a JMP automation script."""

    filename: str
    nrows: int
    ncols: int
    column_names: list[str]
    status_path: str
    script_path: str


class JSLAutomationError(RuntimeError):
    """Raised when JMP automation does not produce a successful status."""


def find_jmp_executable(jmp_executable: str | os.PathLike[str] | None = None) -> str | None:
    """
    Locate a JMP executable for optional JSL automation.

    Resolution order:
    1. Explicit ``jmp_executable`` argument.
    2. ``JMPIO_JMP_EXE`` environment variable.
    3. Common Windows JMP install paths.
    4. ``jmp`` or ``jmp.exe`` on PATH.
    """
    candidates: list[str | os.PathLike[str]] = []
    if jmp_executable:
        candidates.append(jmp_executable)

    env_path = os.environ.get("JMPIO_JMP_EXE")
    if env_path:
        candidates.append(env_path)

    candidates.extend(DEFAULT_JMP_EXECUTABLES)

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)

    return shutil.which("jmp") or shutil.which("jmp.exe")


def write_jmp_with_jsl(
    df: pd.DataFrame,
    filename: str | os.PathLike[str],
    *,
    jmp_executable: str | os.PathLike[str] | None = None,
    timeout: float = 60,
    keep_script: bool = False,
) -> JSLResult:
    """
    Write a JMP file by automating a local SAS JMP installation with JSL.

    This is an optional fallback path. It writes the DataFrame to a temporary
    CSV, asks JMP to open that CSV and save a native JMP table, then moves the
    native output into place after the script reports success.
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jmpio-jsl-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        csv_path = temp_dir / "input.csv"
        native_output_path = temp_dir / "output.jmp"
        status_path = temp_dir / "status.txt"
        script_path = temp_dir / "write.jsl"

        df.to_csv(csv_path, index=False, na_rep="")
        script_path.write_text(
            _build_write_jsl(csv_path, native_output_path, status_path),
            encoding="utf-8",
        )

        result = _run_jsl_script(
            script_path,
            status_path,
            native_output_path,
            jmp_executable=jmp_executable,
            timeout=timeout,
        )

        if not native_output_path.exists():
            raise JSLAutomationError(
                f"JMP automation reported success but did not create {native_output_path}"
            )

        os.replace(native_output_path, output_path)

        if keep_script:
            kept_script = output_path.with_suffix(output_path.suffix + ".write.jsl")
            kept_status = output_path.with_suffix(output_path.suffix + ".write.status.txt")
            shutil.copy2(script_path, kept_script)
            shutil.copy2(status_path, kept_status)
            return JSLResult(
                filename=str(output_path),
                nrows=result.nrows,
                ncols=result.ncols,
                column_names=result.column_names,
                status_path=str(kept_status),
                script_path=str(kept_script),
            )

        return JSLResult(
            filename=str(output_path),
            nrows=result.nrows,
            ncols=result.ncols,
            column_names=result.column_names,
            status_path=result.status_path,
            script_path=result.script_path,
        )


def validate_jmp_with_jsl(
    filename: str | os.PathLike[str],
    *,
    jmp_executable: str | os.PathLike[str] | None = None,
    timeout: float = 60,
    keep_script: bool = False,
) -> JSLResult:
    """
    Validate that a local SAS JMP installation can open a JMP file.

    The generated JSL script opens the target invisibly and writes row count,
    column count, and column names to a status file.
    """
    target_path = Path(filename)
    if not target_path.exists():
        raise FileNotFoundError(target_path)

    with tempfile.TemporaryDirectory(prefix="jmpio-jsl-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        status_path = temp_dir / "status.txt"
        script_path = temp_dir / "validate.jsl"
        script_path.write_text(
            _build_validate_jsl(target_path, status_path),
            encoding="utf-8",
        )

        result = _run_jsl_script(
            script_path,
            status_path,
            target_path,
            jmp_executable=jmp_executable,
            timeout=timeout,
        )

        if keep_script:
            kept_script = target_path.with_suffix(target_path.suffix + ".validate.jsl")
            kept_status = target_path.with_suffix(target_path.suffix + ".validate.status.txt")
            shutil.copy2(script_path, kept_script)
            shutil.copy2(status_path, kept_status)
            return JSLResult(
                filename=str(target_path),
                nrows=result.nrows,
                ncols=result.ncols,
                column_names=result.column_names,
                status_path=str(kept_status),
                script_path=str(kept_script),
            )

        return result


def _run_jsl_script(
    script_path: Path,
    status_path: Path,
    target_path: Path,
    *,
    jmp_executable: str | os.PathLike[str] | None,
    timeout: float,
) -> JSLResult:
    executable = find_jmp_executable(jmp_executable)
    if executable is None:
        raise FileNotFoundError(
            "Could not locate JMP. Pass jmp_executable=... or set JMPIO_JMP_EXE."
        )

    process = subprocess.Popen(
        [executable, str(script_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + timeout
    last_status = ""

    while time.monotonic() < deadline:
        if status_path.exists():
            last_status = status_path.read_text(encoding="utf-8", errors="replace")
            lines = last_status.splitlines()
            if lines and lines[0] in {"OK", "ERROR"}:
                break

        if process.poll() is not None:
            break

        time.sleep(0.25)

    if process.poll() is None and status_path.exists():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    if not status_path.exists():
        raise JSLAutomationError(
            f"JMP automation did not write a status file for {target_path}"
        )

    status = status_path.read_text(encoding="utf-8", errors="replace")
    if status.startswith("START\n"):
        raise JSLAutomationError(
            f"JMP automation started but did not finish for {target_path}"
        )

    return _parse_status(status, target_path, status_path, script_path)


def _parse_status(
    status: str,
    target_path: Path,
    status_path: Path,
    script_path: Path,
) -> JSLResult:
    lines = status.splitlines()
    if not lines:
        raise JSLAutomationError(f"JMP automation wrote an empty status for {target_path}")

    if lines[0] == "ERROR":
        message = lines[1] if len(lines) > 1 else "unknown JMP automation error"
        raise JSLAutomationError(message)

    if lines[0] != "OK" or len(lines) < 3:
        raise JSLAutomationError(f"Unexpected JMP automation status for {target_path}: {status!r}")

    try:
        nrows = int(lines[1])
        ncols = int(lines[2])
    except ValueError as exc:
        raise JSLAutomationError(f"Invalid JMP automation status for {target_path}: {status!r}") from exc

    column_names = lines[3:]
    if len(column_names) != ncols:
        raise JSLAutomationError(
            f"JMP reported {ncols} columns but returned {len(column_names)} names"
        )

    return JSLResult(
        filename=str(target_path),
        nrows=nrows,
        ncols=ncols,
        column_names=column_names,
        status_path=str(status_path),
        script_path=str(script_path),
    )


def _build_write_jsl(
    csv_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    status_path: str | os.PathLike[str],
) -> str:
    return f"""Names Default To Here( 1 );

status_path = {_jsl_string(status_path)};
csv_path = {_jsl_string(csv_path)};
output_path = {_jsl_string(output_path)};

Save Text File( status_path, "START\\!N" || output_path );

Try(
\tdt = Open( csv_path, Invisible );
\tdt << Save( output_path );
\t{_status_jsl()}
\tClose( dt, No Save );
,
\tSave Text File( status_path, "ERROR\\!N" || Char( exception_msg ) );
);

Quit();
"""


def _build_validate_jsl(
    target_path: str | os.PathLike[str],
    status_path: str | os.PathLike[str],
) -> str:
    return f"""Names Default To Here( 1 );

status_path = {_jsl_string(status_path)};
target_path = {_jsl_string(target_path)};

Save Text File( status_path, "START\\!N" || target_path );

Try(
\tdt = Open( target_path, Invisible );
\t{_status_jsl()}
\tClose( dt, No Save );
,
\tSave Text File( status_path, "ERROR\\!N" || Char( exception_msg ) );
);

Quit();
"""


def _status_jsl() -> str:
    return """names = dt << Get Column Names( String );
status = "OK\\!N" || Char( N Rows( dt ) ) || "\\!N" || Char( N Cols( dt ) );
For( i = 1, i <= N Items( names ), i++,
\tstatus ||= "\\!N" || names[i];
);
Save Text File( status_path, status );"""


def _jsl_string(value: str | os.PathLike[str]) -> str:
    path = str(value).replace("\\", "/")
    return '"' + path.replace('"', '\\"') + '"'
