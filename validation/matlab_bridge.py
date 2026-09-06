"""Portable MATLAB ``-batch`` invocation for the SPM12 validation bridge.

Why this module exists
----------------------
The obvious way to drive SPM12 is to join several MATLAB statements into one
``-batch`` string::

    matlab -batch "cd('/path'); setenv('X','y'); run_my_script"

That works on Windows and **fails on Linux**. The Linux ``matlab`` launcher is a
shell script that re-quotes its arguments through an ``eval`` (R2024b, line
1790); a multi-statement string containing single quotes and parentheses is
re-split by the shell and the launcher dies with ``syntax error near unexpected
token '('``. DCCN additionally interposes a Slurm wrapper at
``/opt/cluster/bin/slurm/matlab`` which forwards to that same launcher.

The portable form passes **only single tokens** on the command line:

* the working directory via ``-sd`` rather than a ``cd(...)`` statement,
* the script as a bare name via ``-batch <name>``,
* every parameter through the **child environment**, which every ``.m`` file in
  ``validation/matlab_scripts/`` already reads with ``getenv``.

This is not a workaround -- it is the form that behaves identically on the
workstation and on the cluster, which is what makes the SPM parity tests
portable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from config import MATLAB_PATH, SPM12_PATH

MATLAB_SCRIPTS_DIR = Path(__file__).resolve().parent / "matlab_scripts"


def build_matlab_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Child environment for a MATLAB subprocess.

    Guarantees ``SPM12_PATH`` is set from :data:`config.SPM12_PATH` so the
    ``.m`` files resolve SPM12 identically on every machine, then layers any
    caller-supplied variables on top.

    Parameters
    ----------
    extra : dict[str, str] or None
        Additional environment variables, e.g. ``DCM_INPUT_PATH``.

    Returns
    -------
    dict[str, str]
        A copy of the current environment with the additions applied.
    """
    env = dict(os.environ)
    env.setdefault("SPM12_PATH", str(SPM12_PATH))
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run_matlab_script(
    script_name: str,
    env_vars: dict[str, str] | None = None,
    *,
    timeout: int = 600,
    scripts_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one ``validation/matlab_scripts`` ``.m`` file under ``-batch``.

    Parameters
    ----------
    script_name : str
        Script name WITHOUT the ``.m`` extension, e.g.
        ``"run_spm_spectral_dcm_csd_injected"``. Must be a bare identifier --
        anything with spaces, quotes or semicolons defeats the whole point of
        this module and is rejected.
    env_vars : dict[str, str] or None
        Parameters for the script, read on the MATLAB side via ``getenv``.
    timeout : int
        Subprocess timeout in seconds.
    scripts_dir : pathlib.Path or None
        Directory holding the ``.m`` file. Defaults to
        ``validation/matlab_scripts``.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process, with ``stdout``/``stderr`` captured as text.

    Raises
    ------
    ValueError
        If ``script_name`` is not a bare identifier.
    """
    if not script_name.isidentifier():
        msg = (
            f"script_name must be a bare MATLAB identifier (no path, extension, "
            f"quotes or statements); got {script_name!r}. Pass parameters via "
            f"env_vars instead -- see this module's docstring."
        )
        raise ValueError(msg)

    directory = scripts_dir or MATLAB_SCRIPTS_DIR
    return subprocess.run(
        [str(MATLAB_PATH), "-sd", str(directory), "-batch", script_name],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=build_matlab_env(env_vars),
        check=False,
    )
