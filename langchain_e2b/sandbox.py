"""E2B sandbox backend implementation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
from typing import TYPE_CHECKING, TypeVar

import e2b
from deepagents.backends.protocol import (
    ASYNC_GREP_TIMEOUT,
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import (
    _EDIT_COMMAND_TEMPLATE,
    _EDIT_INLINE_MAX_BYTES,
    _EDIT_TMPFILE_TEMPLATE,
    _GLOB_COMMAND_TEMPLATE,
    _READ_COMMAND_TEMPLATE,
    _WRITE_CHECK_TEMPLATE,
    BaseSandbox,
)
from deepagents.backends.utils import _get_file_type

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

DEFAULT_WORKDIR = "/home/user"
TIMEOUT_EXIT_CODE = 124

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    output = stdout or ""
    if stderr:
        output += "\n" + stderr if output else stderr
    return output


def _run_async_from_sync(factory: Callable[[], Coroutine[object, object, T]]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    msg = (
        "AsyncE2BSandbox sync methods cannot be called from a running event "
        "loop. Use the async method variant instead."
    )
    raise RuntimeError(msg)


class E2BSandbox(BaseSandbox):
    """Sandbox backend that operates on an existing E2B sandbox."""

    def __init__(
        self,
        *,
        sandbox: e2b.Sandbox,
        workdir: str = DEFAULT_WORKDIR,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend wrapping an existing E2B sandbox.

        Args:
            sandbox: Existing E2B sandbox instance to wrap.
            workdir: Working directory for command execution.
            timeout: Default command timeout in seconds when `execute()` is
                called without an explicit `timeout`.
        """
        self._sandbox = sandbox
        self._workdir = workdir
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return the E2B sandbox id."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for this command.

                If None, uses the backend's default timeout.

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.

        Raises:
            ValueError: If `timeout` is negative.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout < 0:
            msg = f"timeout must be non-negative, got {effective_timeout}"
            raise ValueError(msg)

        try:
            result = self._sandbox.commands.run(
                command,
                cwd=self._workdir,
                timeout=effective_timeout,
            )
        except e2b.CommandExitException as exc:
            return ExecuteResponse(
                output=_combine_output(exc.stdout, exc.stderr),
                exit_code=exc.exit_code,
                truncated=False,
            )
        except e2b.TimeoutException:
            return ExecuteResponse(
                output=f"Command timed out after {effective_timeout} seconds",
                exit_code=TIMEOUT_EXIT_CODE,
                truncated=False,
            )
        except e2b.SandboxException as exc:
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
                truncated=False,
            )

        return ExecuteResponse(
            output=_combine_output(result.stdout, result.stderr),
            exit_code=result.exit_code,
            truncated=False,
        )

    def _read_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, content=None, error="invalid_path")

        try:
            info = self._sandbox.files.get_info(path)
            if info.type == e2b.FileType.DIR:
                return FileDownloadResponse(
                    path=path,
                    content=None,
                    error="is_directory",
                )
            content = bytes(self._sandbox.files.read(path, format="bytes"))
            return FileDownloadResponse(path=path, content=content, error=None)
        except e2b.FileNotFoundException:
            return FileDownloadResponse(path=path, content=None, error="file_not_found")
        except e2b.InvalidArgumentException:
            return FileDownloadResponse(path=path, content=None, error="invalid_path")
        except PermissionError:
            return FileDownloadResponse(
                path=path,
                content=None,
                error="permission_denied",
            )

    def _write_file(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")

        error: str | None = None
        try:
            info = self._sandbox.files.get_info(path)
            if info.type == e2b.FileType.DIR:
                error = "is_directory"
        except e2b.FileNotFoundException:
            pass
        except e2b.InvalidArgumentException:
            error = "invalid_path"
        except PermissionError:
            error = "permission_denied"

        if error is None:
            try:
                self._sandbox.files.write(path, content)
            except e2b.FileNotFoundException:
                error = "file_not_found"
            except e2b.InvalidArgumentException:
                error = "invalid_path"
            except PermissionError:
                error = "permission_denied"

        return FileUploadResponse(path=path, error=error)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.

        Args:
            paths: Absolute sandbox file paths to download.

        Returns:
            Download responses in the same order as `paths`.
        """
        return [self._read_file(path) for path in paths]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox.

        Args:
            files: `(path, content)` pairs to write.

        Returns:
            Upload responses in the same order as `files`.
        """
        return [self._write_file(path, content) for path, content in files]


class AsyncE2BSandbox(BaseSandbox):
    """Sandbox backend that operates on an existing async E2B sandbox."""

    def __init__(
        self,
        *,
        sandbox: e2b.AsyncSandbox,
        workdir: str = DEFAULT_WORKDIR,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend wrapping an existing async E2B sandbox.

        Args:
            sandbox: Existing async E2B sandbox instance to wrap.
            workdir: Working directory for command execution.
            timeout: Default command timeout in seconds when `aexecute()` is
                called without an explicit `timeout`.
        """
        self._sandbox = sandbox
        self._workdir = workdir
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return the E2B sandbox id."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Synchronously execute a command using the async E2B sandbox.

        This bridge is only intended for sync callers that are not already
        inside an event loop. Async callers should use `aexecute()`.
        """
        return _run_async_from_sync(lambda: self.aexecute(command, timeout=timeout))

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox asynchronously.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for this command.

                If None, uses the backend's default timeout.

        Returns:
            ExecuteResponse containing output, exit code, and truncation flag.

        Raises:
            ValueError: If `timeout` is negative.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout < 0:
            msg = f"timeout must be non-negative, got {effective_timeout}"
            raise ValueError(msg)

        try:
            result = await self._sandbox.commands.run(
                command,
                cwd=self._workdir,
                timeout=effective_timeout,
            )
        except e2b.CommandExitException as exc:
            return ExecuteResponse(
                output=_combine_output(exc.stdout, exc.stderr),
                exit_code=exc.exit_code,
                truncated=False,
            )
        except e2b.TimeoutException:
            return ExecuteResponse(
                output=f"Command timed out after {effective_timeout} seconds",
                exit_code=TIMEOUT_EXIT_CODE,
                truncated=False,
            )
        except e2b.SandboxException as exc:
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
                truncated=False,
            )

        return ExecuteResponse(
            output=_combine_output(result.stdout, result.stderr),
            exit_code=result.exit_code,
            truncated=False,
        )

    async def _aread_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, content=None, error="invalid_path")

        try:
            info = await self._sandbox.files.get_info(path)
            if info.type == e2b.FileType.DIR:
                return FileDownloadResponse(
                    path=path,
                    content=None,
                    error="is_directory",
                )
            content = bytes(await self._sandbox.files.read(path, format="bytes"))
            return FileDownloadResponse(path=path, content=content, error=None)
        except e2b.FileNotFoundException:
            return FileDownloadResponse(path=path, content=None, error="file_not_found")
        except e2b.InvalidArgumentException:
            return FileDownloadResponse(path=path, content=None, error="invalid_path")
        except PermissionError:
            return FileDownloadResponse(
                path=path,
                content=None,
                error="permission_denied",
            )

    async def _awrite_file(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")

        error: str | None = None
        try:
            info = await self._sandbox.files.get_info(path)
            if info.type == e2b.FileType.DIR:
                error = "is_directory"
        except e2b.FileNotFoundException:
            pass
        except e2b.InvalidArgumentException:
            error = "invalid_path"
        except PermissionError:
            error = "permission_denied"

        if error is None:
            try:
                await self._sandbox.files.write(path, content)
            except e2b.FileNotFoundException:
                error = "file_not_found"
            except e2b.InvalidArgumentException:
                error = "invalid_path"
            except PermissionError:
                error = "permission_denied"

        return FileUploadResponse(path=path, error=error)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Synchronously download files using the async E2B sandbox."""
        return _run_async_from_sync(lambda: self.adownload_files(paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox asynchronously.

        Args:
            paths: Absolute sandbox file paths to download.

        Returns:
            Download responses in the same order as `paths`.
        """
        return [await self._aread_file(path) for path in paths]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Synchronously upload files using the async E2B sandbox."""
        return _run_async_from_sync(lambda: self.aupload_files(files))

    async def aupload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files into the sandbox asynchronously.

        Args:
            files: `(path, content)` pairs to write.

        Returns:
            Upload responses in the same order as `files`.
        """
        return [await self._awrite_file(path, content) for path, content in files]

    async def als(self, path: str) -> LsResult:
        """Structured async listing with file metadata using os.scandir."""
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
        cmd = f"""python3 -c "
import os
import json
import base64

path = base64.b64decode('{path_b64}').decode('utf-8')

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': os.path.join(path, entry.name),
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    print(json.dumps({{'error': 'path_not_found'}}))
except NotADirectoryError:
    print(json.dumps({{'error': 'not_a_directory'}}))
except PermissionError:
    print(json.dumps({{'error': 'permission_denied'}}))
" 2>/dev/null"""

        result = await self.aexecute(cmd)

        file_infos: list[FileInfo] = []
        error: str | None = None
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "error" in data:
                error = data["error"]
                continue
            file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})

        if error is not None:
            return LsResult(entries=None, error=f"Path '{path}': {error}")
        return LsResult(entries=file_infos)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content asynchronously with server-side line pagination."""
        file_type = _get_file_type(file_path)
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")

        cmd = _READ_COMMAND_TEMPLATE.format(
            path_b64=path_b64,
            file_type=file_type,
            offset=int(offset),
            limit=int(limit),
        )
        result = await self.aexecute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            detail = output[:200] if output else "(empty)"
            return ReadResult(
                error=f"File '{file_path}': unexpected server response: {detail}"
            )

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            return ReadResult(
                error=f"File '{file_path}': unexpected server response: {detail}"
            )

        if "error" in data:
            return ReadResult(error=f"File '{file_path}': {data['error']}")

        return ReadResult(
            file_data=FileData(
                content=data["content"],
                encoding=data.get("encoding", "utf-8"),
            )
        )

    async def _awrite_preflight(self, file_path: str) -> WriteResult | None:
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")
        check_cmd = _WRITE_CHECK_TEMPLATE.format(path_b64=path_b64)
        result = await self.aexecute(check_cmd)
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)
        return None

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file asynchronously, failing if it already exists."""
        preflight_error = await self._awrite_preflight(file_path)
        if preflight_error is not None:
            return preflight_error

        responses = await self.aupload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            msg = (
                "Responses was expected to return 1 result, but it returned "
                f"{len(responses)} with type {type(responses)}"
            )
            raise AssertionError(msg)
        response = responses[0]
        if response.error:
            return WriteResult(
                error=f"Failed to write file '{file_path}': {response.error}"
            )

        return WriteResult(path=file_path)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file asynchronously by replacing exact string occurrences."""
        payload_size = len(old_string.encode("utf-8")) + len(new_string.encode("utf-8"))

        if payload_size <= _EDIT_INLINE_MAX_BYTES:
            return await self._aedit_inline(
                file_path,
                old_string,
                new_string,
                replace_all,
            )

        return await self._aedit_via_upload(
            file_path,
            old_string,
            new_string,
            replace_all,
        )

    async def _aedit_inline(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        payload = json.dumps(
            {
                "path": file_path,
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
            }
        )
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
        result = await self.aexecute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            detail = output[:200] if output else "(empty)"
            error = (
                f"Error editing file '{file_path}': unexpected server response: "
                f"{detail}"
            )
            return EditResult(error=error)

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            error = (
                f"Error editing file '{file_path}': unexpected server response: "
                f"{detail}"
            )
            return EditResult(error=error)

        if "error" in data:
            return self._map_edit_error(data["error"], file_path, old_string)

        return EditResult(
            path=file_path,
            occurrences=data.get("count", 1),
        )

    async def _aedit_via_upload(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        uid = base64.b32encode(os.urandom(10)).decode("ascii").lower()
        old_tmp = f"/tmp/.deepagents_edit_{uid}_old"  # noqa: S108
        new_tmp = f"/tmp/.deepagents_edit_{uid}_new"  # noqa: S108

        resps = await self.aupload_files(
            [
                (old_tmp, old_string.encode("utf-8")),
                (new_tmp, new_string.encode("utf-8")),
            ]
        )
        if len(resps) < 2:  # noqa: PLR2004
            return EditResult(
                error=f"Error editing file '{file_path}': upload returned no response"
            )
        for response in resps:
            if response.error:
                return EditResult(
                    error=f"Error editing file '{file_path}': {response.error}"
                )

        cmd = _EDIT_TMPFILE_TEMPLATE.format(
            old_path_b64=base64.b64encode(old_tmp.encode("utf-8")).decode("ascii"),
            new_path_b64=base64.b64encode(new_tmp.encode("utf-8")).decode("ascii"),
            target_b64=base64.b64encode(file_path.encode("utf-8")).decode("ascii"),
            replace_all=replace_all,
        )
        result = await self.aexecute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            cleanup = await self.aexecute(
                f"rm -f {shlex.quote(old_tmp)} {shlex.quote(new_tmp)}"
            )
            if cleanup.exit_code != 0:
                logger.warning(
                    "Failed to clean up temp files for edit %s: %s",
                    file_path,
                    cleanup.output[:200],
                )
            detail = output[:200] if output else "(empty)"
            error = (
                f"Error editing file '{file_path}': unexpected server response: "
                f"{detail}"
            )
            return EditResult(error=error)

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            error = (
                f"Error editing file '{file_path}': unexpected server response: "
                f"{detail}"
            )
            return EditResult(error=error)

        if "error" in data:
            return self._map_edit_error(data["error"], file_path, old_string)

        return EditResult(
            path=file_path,
            occurrences=data.get("count", 1),
        )

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search file contents asynchronously for a literal string."""
        try:
            return await asyncio.wait_for(
                self._agrep(pattern, path=path, glob=glob),
                timeout=ASYNC_GREP_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "agrep timed out after %ds (pattern=%r, path=%r, glob=%r)",
                ASYNC_GREP_TIMEOUT,
                pattern,
                path,
                glob,
            )
            return GrepResult(
                error=(
                    f"Error: grep timed out after {ASYNC_GREP_TIMEOUT}s. "
                    "Try a more specific pattern or a narrower path."
                ),
            )

    async def _agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        search_path = shlex.quote(path or ".")
        grep_opts = "-rHnFZ"
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include={shlex.quote(glob)}"
        pattern_escaped = shlex.quote(pattern)

        cmd = (
            f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} "
            f"{search_path} 2>/dev/null || true"
        )
        result = await self.aexecute(cmd)

        output = result.output.rstrip("\n")
        if result.exit_code is not None and result.exit_code != 0:
            detail = output.strip() if output else f"exit code {result.exit_code}"
            return GrepResult(error=f"Path '{path or '.'}': {detail}")
        if not output:
            return GrepResult(matches=[])

        matches: list[GrepMatch] = []
        parse_error: str | None = None
        for line in output.split("\n"):
            parts = line.split("\0", 1)
            if len(parts) != 2:  # noqa: PLR2004
                parse_error = line
                continue
            line_parts = parts[1].split(":", 1)
            if len(line_parts) != 2:  # noqa: PLR2004
                parse_error = line
                continue
            try:
                matches.append(
                    {
                        "path": parts[0],
                        "line": int(line_parts[0]),
                        "text": line_parts[1],
                    }
                )
            except ValueError:
                parse_error = line

        if parse_error is not None and not matches:
            return GrepResult(error=f"Path '{path or '.'}': {parse_error}")

        return GrepResult(matches=matches)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Structured async glob matching returning `GlobResult`."""
        search_path = path or "/"
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(search_path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(
            path_b64=path_b64,
            pattern_b64=pattern_b64,
        )
        result = await self.aexecute(cmd)

        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])

        file_infos: list[FileInfo] = []
        error: str | None = None
        for line in output.split("\n"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "error" in data:
                error = data["error"]
                continue
            file_infos.append(
                {
                    "path": data["path"],
                    "is_dir": data["is_dir"],
                }
            )

        if error is not None:
            return GlobResult(matches=None, error=f"Path '{search_path}': {error}")
        return GlobResult(matches=file_infos)
