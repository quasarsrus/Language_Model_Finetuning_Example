import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: list[str]) -> None:
    """Execute shell commands with `pytest` and `sh` package.

    params:
        command: A list of shell commands as strings.
    """
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(reason=msg)
