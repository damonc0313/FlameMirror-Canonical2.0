"""Sandbox execution environment."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional

SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "range": range,
    "len": len,
    "sum": sum,
    "enumerate": enumerate,
}


@dataclass(slots=True)
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    exception: Optional[BaseException] = None


class Sandbox:
    """A minimal sandbox for executing generated code."""

    def __init__(self, allowed_builtins: Optional[Dict[str, Any]] = None) -> None:
        self.allowed_builtins = dict(SAFE_BUILTINS)
        if allowed_builtins:
            self.allowed_builtins.update(allowed_builtins)

    def execute(self, code: str, *, globals_dict: Optional[Dict[str, Any]] = None) -> SandboxResult:
        local_globals = {"__builtins__": self.allowed_builtins}
        if globals_dict is not None:
            local_globals.update(globals_dict)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                exec(code, local_globals)
        except BaseException as exc:  # noqa: BLE001 - we intentionally catch everything
            return SandboxResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                exception=exc,
            )
        if globals_dict is not None:
            globals_dict.update({k: v for k, v in local_globals.items() if k != "__builtins__"})
        return SandboxResult(
            success=True,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
        )

    def evaluate_function(
        self,
        code: str,
        function_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> SandboxResult:
        globals_dict: Dict[str, Any] = {}
        execution = self.execute(code, globals_dict=globals_dict)
        if not execution.success:
            return execution
        func = globals_dict.get(function_name)
        if not callable(func):
            return SandboxResult(
                success=False,
                stdout=execution.stdout,
                stderr="Function not defined",
                exception=AttributeError(function_name),
            )
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                result = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            return SandboxResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                exception=exc,
            )
        globals_dict["result"] = result
        return SandboxResult(
            success=True,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
        )
