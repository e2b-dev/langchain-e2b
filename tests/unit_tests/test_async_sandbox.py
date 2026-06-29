from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import e2b
import pytest

from langchain_e2b import AsyncE2BSandbox

if TYPE_CHECKING:
    from e2b import AsyncSandbox

TEST_DIR_PATH = "/home/user/data"
TEST_FILE_PATH = "/home/user/file.txt"
TEST_TIMEOUT = 7
TIMEOUT_EXIT_CODE = 124


@dataclass
class FakeEntryInfo:
    type: e2b.FileType


@dataclass
class FakeCommandResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class FakeAsyncFiles:
    entries: dict[str, tuple[e2b.FileType, bytes]] = field(default_factory=dict)
    get_info_exc: BaseException | None = None
    write_exc: BaseException | None = None

    async def get_info(self, path: str) -> FakeEntryInfo:
        if self.get_info_exc is not None:
            raise self.get_info_exc
        if path not in self.entries:
            raise e2b.FileNotFoundException(path)

        file_type, _ = self.entries[path]
        return FakeEntryInfo(type=file_type)

    async def read(
        self,
        path: str,
        *,
        format: Literal["bytes"] = "bytes",  # noqa: A002
    ) -> bytearray:
        assert format == "bytes"
        return bytearray(self.entries[path][1])

    async def write(self, path: str, data: bytes) -> None:
        if self.write_exc is not None:
            raise self.write_exc
        self.entries[path] = (e2b.FileType.FILE, data)


@dataclass
class FakeAsyncCommands:
    result: FakeCommandResult | None = None
    exc: BaseException | None = None
    cwd: str | None = None
    timeout: int | None = None

    async def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> FakeCommandResult:
        self.cwd = cwd
        self.timeout = timeout
        if self.exc is not None:
            raise self.exc
        assert command
        assert self.result is not None
        return self.result


@dataclass
class FakeAsyncSandbox:
    commands: FakeAsyncCommands
    files: FakeAsyncFiles
    sandbox_id: str = "sbx_async_test"


def _backend(
    *,
    commands: FakeAsyncCommands | None = None,
    files: FakeAsyncFiles | None = None,
    workdir: str = "/home/user",
    timeout: int = 30 * 60,
) -> AsyncE2BSandbox:
    fake = FakeAsyncSandbox(
        commands=commands
        or FakeAsyncCommands(
            result=FakeCommandResult(stdout="", stderr="", exit_code=0)
        ),
        files=files or FakeAsyncFiles(),
    )
    return AsyncE2BSandbox(
        sandbox=cast("AsyncSandbox", fake),
        workdir=workdir,
        timeout=timeout,
    )


def test_id_returns_e2b_async_sandbox_id() -> None:
    backend = _backend()

    assert backend.id == "sbx_async_test"


def test_execute_bridges_to_async_command_without_running_loop() -> None:
    commands = FakeAsyncCommands(
        result=FakeCommandResult(stdout="hello\n", stderr="", exit_code=0)
    )
    backend = _backend(commands=commands, workdir="/workspace", timeout=TEST_TIMEOUT)

    result = backend.execute("echo hello")

    assert result.output == "hello\n"
    assert result.exit_code == 0
    assert commands.cwd == "/workspace"
    assert commands.timeout == TEST_TIMEOUT


async def test_aexecute_success_uses_workdir_and_timeout() -> None:
    commands = FakeAsyncCommands(
        result=FakeCommandResult(stdout="hello\n", stderr="", exit_code=0)
    )
    backend = _backend(commands=commands, workdir="/workspace", timeout=TEST_TIMEOUT)

    result = await backend.aexecute("echo hello")

    assert result.output == "hello\n"
    assert result.exit_code == 0
    assert result.truncated is False
    assert commands.cwd == "/workspace"
    assert commands.timeout == TEST_TIMEOUT


async def test_aexecute_combines_stdout_and_stderr() -> None:
    backend = _backend(
        commands=FakeAsyncCommands(
            result=FakeCommandResult(stdout="out", stderr="err", exit_code=0)
        )
    )

    result = await backend.aexecute("echo hello")

    assert result.output == "out\nerr"
    assert result.exit_code == 0


async def test_aexecute_nonzero_exit_returns_response() -> None:
    backend = _backend(
        commands=FakeAsyncCommands(
            exc=e2b.CommandExitException(
                stdout="",
                stderr="boom",
                exit_code=1,
                error="boom",
            )
        )
    )

    result = await backend.aexecute("false")

    assert result.output == "boom"
    assert result.exit_code == 1


async def test_aexecute_timeout_returns_timeout_response() -> None:
    backend = _backend(
        commands=FakeAsyncCommands(exc=e2b.TimeoutException("timed out"))
    )

    result = await backend.aexecute("sleep 10", timeout=5)

    assert result.output == "Command timed out after 5 seconds"
    assert result.exit_code == TIMEOUT_EXIT_CODE


async def test_aexecute_rejects_negative_timeout() -> None:
    backend = _backend()

    with pytest.raises(ValueError, match="timeout must be non-negative"):
        await backend.aexecute("echo hello", timeout=-1)


async def test_adownload_rejects_relative_path() -> None:
    response = (await _backend().adownload_files(["relative.txt"]))[0]

    assert response.error == "invalid_path"
    assert response.content is None


async def test_adownload_missing_file_maps_to_file_not_found() -> None:
    response = (await _backend().adownload_files([TEST_FILE_PATH]))[0]

    assert response.error == "file_not_found"
    assert response.content is None


async def test_adownload_directory_maps_to_is_directory() -> None:
    files = FakeAsyncFiles(entries={TEST_DIR_PATH: (e2b.FileType.DIR, b"")})
    response = (await _backend(files=files).adownload_files([TEST_DIR_PATH]))[0]

    assert response.error == "is_directory"
    assert response.content is None


async def test_adownload_invalid_argument_maps_to_invalid_path() -> None:
    files = FakeAsyncFiles(get_info_exc=e2b.InvalidArgumentException("invalid path"))
    response = (await _backend(files=files).adownload_files([TEST_FILE_PATH]))[0]

    assert response.error == "invalid_path"
    assert response.content is None


async def test_aupload_rejects_relative_path() -> None:
    response = (await _backend().aupload_files([("relative.txt", b"hello")]))[0]

    assert response.error == "invalid_path"


async def test_aupload_existing_directory_maps_to_is_directory() -> None:
    files = FakeAsyncFiles(entries={TEST_DIR_PATH: (e2b.FileType.DIR, b"")})
    response = (await _backend(files=files).aupload_files([(TEST_DIR_PATH, b"hello")]))[
        0
    ]

    assert response.error == "is_directory"


async def test_aupload_invalid_argument_maps_to_invalid_path() -> None:
    files = FakeAsyncFiles(write_exc=e2b.InvalidArgumentException("invalid path"))
    responses = await _backend(files=files).aupload_files([(TEST_FILE_PATH, b"hello")])
    response = responses[0]

    assert response.error == "invalid_path"


async def test_aupload_and_adownload_round_trip() -> None:
    files = FakeAsyncFiles()
    backend = _backend(files=files)

    upload = (await backend.aupload_files([(TEST_FILE_PATH, b"hello")]))[0]
    download = (await backend.adownload_files([TEST_FILE_PATH]))[0]

    assert upload.error is None
    assert download.error is None
    assert download.content == b"hello"


async def test_als_parses_json_entries() -> None:
    backend = _backend(
        commands=FakeAsyncCommands(
            result=FakeCommandResult(
                stdout='{"path": "/home/user/file.txt", "is_dir": false}\n',
                stderr="",
                exit_code=0,
            )
        )
    )

    result = await backend.als("/home/user")

    assert result.error is None
    assert result.entries == [{"path": "/home/user/file.txt", "is_dir": False}]


async def test_aread_parses_json_file_data() -> None:
    backend = _backend(
        commands=FakeAsyncCommands(
            result=FakeCommandResult(
                stdout='{"encoding": "utf-8", "content": "hello"}\n',
                stderr="",
                exit_code=0,
            )
        )
    )

    result = await backend.aread(TEST_FILE_PATH)

    assert result.error is None
    assert result.file_data == {"encoding": "utf-8", "content": "hello"}


async def test_awrite_runs_preflight_and_uploads_content() -> None:
    files = FakeAsyncFiles()
    backend = _backend(files=files)

    result = await backend.awrite(TEST_FILE_PATH, "hello")

    assert result.error is None
    assert result.path == TEST_FILE_PATH
    assert files.entries[TEST_FILE_PATH] == (e2b.FileType.FILE, b"hello")


async def test_agrep_parses_matches() -> None:
    backend = _backend(
        commands=FakeAsyncCommands(
            result=FakeCommandResult(
                stdout="src/app.py\00012:needle here\n",
                stderr="",
                exit_code=0,
            )
        )
    )

    result = await backend.agrep("needle", path="/workspace")

    assert result.error is None
    assert result.matches == [{"path": "src/app.py", "line": 12, "text": "needle here"}]


async def test_aglob_parses_json_matches() -> None:
    backend = _backend(
        commands=FakeAsyncCommands(
            result=FakeCommandResult(
                stdout='{"path": "src/app.py", "is_dir": false}\n',
                stderr="",
                exit_code=0,
            )
        )
    )

    result = await backend.aglob("**/*.py", path="/workspace")

    assert result.error is None
    assert result.matches == [{"path": "src/app.py", "is_dir": False}]


async def test_sync_methods_reject_running_loop() -> None:
    backend = _backend()

    with pytest.raises(RuntimeError, match="Use the async method variant"):
        backend.execute("echo hello")
