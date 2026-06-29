from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

import httpx
import pytest
from e2b import Sandbox
from e2b.exceptions import SandboxException
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_e2b import AsyncE2BSandbox, E2BSandbox
from langchain_e2b.provider import E2BProvider

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

KILL_ATTEMPTS = 3
KILL_RETRY_DELAY_SECONDS = 1


class TestE2BSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            pytest.skip("Missing secrets for E2B integration test: set E2B_API_KEY")

        template = os.environ.get("E2B_TEMPLATE")
        if template:
            sandbox = Sandbox.create(
                template=template,
                timeout=60 * 60,
                api_key=api_key,
            )
        else:
            sandbox = Sandbox.create(
                timeout=60 * 60,
                api_key=api_key,
            )
        backend = E2BSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            _kill_sandbox(sandbox)


def test_e2b_provider_creates_executes_and_deletes_sandbox() -> None:
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        pytest.skip(
            "Missing secrets for E2B provider integration test: set E2B_API_KEY"
        )

    provider = E2BProvider()
    sandbox_id: str | None = None
    try:
        backend = provider.get_or_create(
            timeout=60 * 60,
            template=os.environ.get("E2B_TEMPLATE"),
            command_timeout=30,
        )
        sandbox_id = backend.id
        result = backend.execute("echo provider-ready", timeout=10)

        assert result.exit_code == 0
        assert "provider-ready" in result.output
    finally:
        if sandbox_id is not None:
            _delete_provider_sandbox(provider, sandbox_id)


async def test_e2b_provider_async_executes_transfers_and_deletes_sandbox() -> None:
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        pytest.skip(
            "Missing secrets for E2B async provider integration test: set E2B_API_KEY"
        )

    provider = E2BProvider()
    sandbox_id: str | None = None
    try:
        backend = await provider.aget_or_create(
            timeout=60 * 60,
            template=os.environ.get("E2B_TEMPLATE"),
            command_timeout=30,
        )
        sandbox_id = backend.id

        assert isinstance(backend, AsyncE2BSandbox)

        result = await backend.aexecute("echo async-provider-ready", timeout=10)
        assert result.exit_code == 0
        assert "async-provider-ready" in result.output

        upload = (await backend.aupload_files([("/home/user/async.txt", b"ok")]))[0]
        assert upload.error is None

        download = (await backend.adownload_files(["/home/user/async.txt"]))[0]
        assert download.error is None
        assert download.content == b"ok"
    finally:
        if sandbox_id is not None:
            await _adelete_provider_sandbox(provider, sandbox_id)


def _kill_sandbox(sandbox: Sandbox) -> None:
    last_error: BaseException | None = None
    for attempt in range(KILL_ATTEMPTS):
        try:
            sandbox.kill()
        except (httpx.HTTPError, SandboxException) as exc:
            last_error = exc
            if attempt + 1 < KILL_ATTEMPTS:
                time.sleep(KILL_RETRY_DELAY_SECONDS)
        else:
            return

    msg = f"Failed to kill E2B sandbox {sandbox.sandbox_id!r}"
    raise RuntimeError(msg) from last_error


def _delete_provider_sandbox(provider: E2BProvider, sandbox_id: str) -> None:
    last_error: BaseException | None = None
    for attempt in range(KILL_ATTEMPTS):
        try:
            provider.delete(sandbox_id=sandbox_id)
        except (httpx.HTTPError, SandboxException) as exc:
            last_error = exc
            if attempt + 1 < KILL_ATTEMPTS:
                time.sleep(KILL_RETRY_DELAY_SECONDS)
        else:
            return

    msg = f"Failed to delete E2B sandbox {sandbox_id!r}"
    raise RuntimeError(msg) from last_error


async def _adelete_provider_sandbox(provider: E2BProvider, sandbox_id: str) -> None:
    last_error: BaseException | None = None
    for attempt in range(KILL_ATTEMPTS):
        try:
            await provider.adelete(sandbox_id=sandbox_id)
        except (httpx.HTTPError, SandboxException) as exc:
            last_error = exc
            if attempt + 1 < KILL_ATTEMPTS:
                await asyncio.sleep(KILL_RETRY_DELAY_SECONDS)
        else:
            return

    msg = f"Failed to delete E2B sandbox {sandbox_id!r}"
    raise RuntimeError(msg) from last_error
