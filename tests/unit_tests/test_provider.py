from __future__ import annotations

from importlib.metadata import entry_points
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import e2b
import pytest
from deepagents_code.integrations.sandbox_config import SandboxConfig
from deepagents_code.integrations.sandbox_provider import SandboxNotFoundError
from deepagents_code.integrations.sandbox_registry import SandboxRegistry

from langchain_e2b import AsyncE2BSandbox, E2BSandbox
from langchain_e2b.provider import (
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_SANDBOX_TIMEOUT,
    E2BProvider,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TEST_COMMAND_TIMEOUT = 42


def _resolver(values: dict[str, str]) -> Callable[[str], str | None]:
    return values.get


def _sandbox(sandbox_id: str = "sbx-test") -> e2b.Sandbox:
    sandbox = MagicMock(sandbox_id=sandbox_id)
    sandbox.commands.run.return_value = SimpleNamespace(
        stdout="",
        stderr="",
        exit_code=0,
    )
    return cast("e2b.Sandbox", sandbox)


def _async_sandbox(sandbox_id: str = "sbx-test") -> e2b.AsyncSandbox:
    sandbox = MagicMock(sandbox_id=sandbox_id)
    sandbox.commands.run = AsyncMock(
        return_value=SimpleNamespace(
            stdout="",
            stderr="",
            exit_code=0,
        )
    )
    return cast("e2b.AsyncSandbox", sandbox)


def test_provider_metadata_does_not_require_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_E2B_API_KEY", raising=False)

    metadata = E2BProvider().metadata

    assert metadata.name == "e2b"
    assert metadata.working_dir == "/home/user"
    assert metadata.supports_sandbox_id is True
    assert metadata.supports_snapshot_name is False
    assert metadata.backend_module == "langchain_e2b"
    assert metadata.install is not None
    assert metadata.install.kind == "package"
    assert metadata.install.name == "langchain-e2b"


def test_entry_point_loads_provider() -> None:
    entries = entry_points(group="deepagents_code.sandbox_providers")
    entry = next(item for item in entries if item.name == "e2b")

    assert entry.value == "langchain_e2b.provider:E2BProvider"
    assert entry.load() is E2BProvider


def test_sandbox_registry_reads_entry_point_metadata_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_E2B_API_KEY", raising=False)
    registry = SandboxRegistry(config=SandboxConfig(), include_entry_points=True)

    metadata = registry.get_metadata("e2b")

    assert metadata is not None
    assert metadata.name == "e2b"
    assert metadata.working_dir == "/home/user"
    assert metadata.backend_module == "langchain_e2b"


def test_provider_prefers_deepagents_code_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "canonical-key")
    monkeypatch.setenv("DEEPAGENTS_CODE_E2B_API_KEY", "prefixed-key")
    sandbox = _sandbox()

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider()
        provider.get_or_create()

    create.assert_called_once_with(
        timeout=DEFAULT_SANDBOX_TIMEOUT,
        api_key="prefixed-key",
    )


def test_empty_deepagents_code_api_key_shadows_canonical_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "canonical-key")
    monkeypatch.setenv("DEEPAGENTS_CODE_E2B_API_KEY", "")
    provider = E2BProvider()

    with pytest.raises(ValueError, match=r"E2B_API_KEY.*DEEPAGENTS_CODE_E2B_API_KEY"):
        provider.get_or_create()


def test_provider_creates_sandbox_with_default_options() -> None:
    sandbox = _sandbox("sbx-new")

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        backend = provider.get_or_create()

    create.assert_called_once_with(
        timeout=DEFAULT_SANDBOX_TIMEOUT,
        api_key="fake-key",
    )
    assert isinstance(backend, E2BSandbox)
    assert backend.id == "sbx-new"

    backend.execute("true")

    cast("MagicMock", sandbox.commands.run).assert_called_once_with(
        "true",
        cwd="/home/user",
        timeout=DEFAULT_COMMAND_TIMEOUT,
    )


async def test_provider_creates_async_sandbox_with_default_options() -> None:
    sandbox = _async_sandbox("sbx-async-new")

    with patch(
        "langchain_e2b.provider.e2b.AsyncSandbox.create",
        new_callable=AsyncMock,
    ) as create:
        create.return_value = sandbox
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        backend = await provider.aget_or_create()

    create.assert_awaited_once_with(
        timeout=DEFAULT_SANDBOX_TIMEOUT,
        api_key="fake-key",
    )
    assert isinstance(backend, AsyncE2BSandbox)
    assert backend.id == "sbx-async-new"

    await backend.aexecute("true")

    cast("AsyncMock", sandbox.commands.run).assert_awaited_once_with(
        "true",
        cwd="/home/user",
        timeout=DEFAULT_COMMAND_TIMEOUT,
    )


def test_provider_creates_sandbox_with_curated_options() -> None:
    sandbox = _sandbox("sbx-template")

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        backend = provider.get_or_create(
            timeout=7200,
            template="custom-template",
            workdir="/workspace",
            command_timeout=TEST_COMMAND_TIMEOUT,
        )

    create.assert_called_once_with(
        template="custom-template",
        timeout=7200,
        api_key="fake-key",
    )
    assert backend.id == "sbx-template"

    backend.execute("true")

    cast("MagicMock", sandbox.commands.run).assert_called_once_with(
        "true",
        cwd="/workspace",
        timeout=TEST_COMMAND_TIMEOUT,
    )


async def test_provider_creates_async_sandbox_with_curated_options() -> None:
    sandbox = _async_sandbox("sbx-async-template")

    with patch(
        "langchain_e2b.provider.e2b.AsyncSandbox.create",
        new_callable=AsyncMock,
    ) as create:
        create.return_value = sandbox
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        backend = await provider.aget_or_create(
            timeout=7200,
            template="custom-template",
            workdir="/workspace",
            command_timeout=TEST_COMMAND_TIMEOUT,
        )

    create.assert_awaited_once_with(
        template="custom-template",
        timeout=7200,
        api_key="fake-key",
    )
    assert backend.id == "sbx-async-template"

    await backend.aexecute("true")

    cast("AsyncMock", sandbox.commands.run).assert_awaited_once_with(
        "true",
        cwd="/workspace",
        timeout=TEST_COMMAND_TIMEOUT,
    )


def test_provider_reads_template_and_timeout_from_environment() -> None:
    sandbox = _sandbox("sbx-template")
    resolve_env_var = _resolver(
        {
            "E2B_API_KEY": "fake-key",
            "E2B_TEMPLATE": "env-template",
            "E2B_SANDBOX_TIMEOUT": "3600",
        }
    )

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider(resolve_env_var=resolve_env_var)
        provider.get_or_create()

    create.assert_called_once_with(
        template="env-template",
        timeout=3600,
        api_key="fake-key",
    )


def test_provider_prefers_deepagents_code_template_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    monkeypatch.setenv("E2B_TEMPLATE", "canonical-template")
    monkeypatch.setenv("DEEPAGENTS_CODE_E2B_TEMPLATE", "prefixed-template")
    monkeypatch.setenv("E2B_SANDBOX_TIMEOUT", "3600")
    monkeypatch.setenv("DEEPAGENTS_CODE_E2B_SANDBOX_TIMEOUT", "7200")
    sandbox = _sandbox("sbx-prefixed")

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider()
        provider.get_or_create()

    create.assert_called_once_with(
        template="prefixed-template",
        timeout=7200,
        api_key="fake-key",
    )


def test_provider_connects_existing_sandbox() -> None:
    sandbox = _sandbox("sbx-existing")

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.connect",
        return_value=sandbox,
    ) as connect:
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        backend = provider.get_or_create(sandbox_id="sbx-existing", timeout=2)

    connect.assert_called_once_with(
        "sbx-existing",
        timeout=2,
        api_key="fake-key",
    )
    assert backend.id == "sbx-existing"


async def test_provider_connects_existing_async_sandbox() -> None:
    sandbox = _async_sandbox("sbx-async-existing")

    with patch(
        "langchain_e2b.provider.e2b.AsyncSandbox.connect",
        new_callable=AsyncMock,
    ) as connect:
        connect.return_value = sandbox
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        backend = await provider.aget_or_create(
            sandbox_id="sbx-async-existing",
            timeout=2,
        )

    connect.assert_awaited_once_with(
        "sbx-async-existing",
        timeout=2,
        api_key="fake-key",
    )
    assert backend.id == "sbx-async-existing"


def test_provider_deletes_sandbox() -> None:
    with patch("langchain_e2b.provider.e2b.Sandbox.kill") as kill:
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        provider.delete(sandbox_id="sbx-delete")

    kill.assert_called_once_with("sbx-delete", api_key="fake-key")


async def test_provider_deletes_async_sandbox() -> None:
    with patch(
        "langchain_e2b.provider.e2b.AsyncSandbox.kill",
        new_callable=AsyncMock,
    ) as kill:
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))
        await provider.adelete(sandbox_id="sbx-async-delete")

    kill.assert_awaited_once_with("sbx-async-delete", api_key="fake-key")


def test_provider_maps_missing_sandbox_to_sandbox_not_found() -> None:
    with patch(
        "langchain_e2b.provider.e2b.Sandbox.connect",
        side_effect=e2b.SandboxNotFoundException("missing"),
    ):
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))

        with pytest.raises(SandboxNotFoundError) as exc_info:
            provider.get_or_create(sandbox_id="sbx-missing")

    assert exc_info.value.args == ("sbx-missing",)


async def test_provider_maps_missing_async_sandbox_to_sandbox_not_found() -> None:
    with patch(
        "langchain_e2b.provider.e2b.AsyncSandbox.connect",
        new_callable=AsyncMock,
    ) as connect:
        connect.side_effect = e2b.SandboxNotFoundException("missing")
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))

        with pytest.raises(SandboxNotFoundError) as exc_info:
            await provider.aget_or_create(sandbox_id="sbx-async-missing")

    assert exc_info.value.args == ("sbx-async-missing",)


def test_provider_maps_missing_delete_to_sandbox_not_found() -> None:
    with patch(
        "langchain_e2b.provider.e2b.Sandbox.kill",
        side_effect=e2b.SandboxNotFoundException("missing"),
    ):
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))

        with pytest.raises(SandboxNotFoundError) as exc_info:
            provider.delete(sandbox_id="sbx-missing")

    assert exc_info.value.args == ("sbx-missing",)


async def test_provider_maps_missing_async_delete_to_sandbox_not_found() -> None:
    with patch(
        "langchain_e2b.provider.e2b.AsyncSandbox.kill",
        new_callable=AsyncMock,
    ) as kill:
        kill.side_effect = e2b.SandboxNotFoundException("missing")
        provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))

        with pytest.raises(SandboxNotFoundError) as exc_info:
            await provider.adelete(sandbox_id="sbx-async-missing")

    assert exc_info.value.args == ("sbx-async-missing",)


def test_provider_requires_api_key() -> None:
    provider = E2BProvider(resolve_env_var=_resolver({}))

    with pytest.raises(ValueError, match=r"E2B_API_KEY.*DEEPAGENTS_CODE_E2B_API_KEY"):
        provider.get_or_create()


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
def test_provider_rejects_invalid_sandbox_timeout(value: str) -> None:
    provider = E2BProvider(
        resolve_env_var=_resolver(
            {
                "E2B_API_KEY": "fake-key",
                "E2B_SANDBOX_TIMEOUT": value,
            }
        )
    )

    with pytest.raises(ValueError, match="timeout"):
        provider.get_or_create()


def test_provider_rejects_invalid_command_timeout() -> None:
    provider = E2BProvider(resolve_env_var=_resolver({"E2B_API_KEY": "fake-key"}))

    with pytest.raises(ValueError, match="command_timeout"):
        provider.get_or_create(command_timeout=-1)


def test_provider_rejects_unsupported_get_options() -> None:
    provider = E2BProvider(resolve_env_var=_resolver({}))

    with pytest.raises(TypeError, match="Received unsupported arguments: metadata"):
        provider.get_or_create(metadata={"purpose": "test"})


def test_provider_rejects_unsupported_delete_options() -> None:
    provider = E2BProvider(resolve_env_var=_resolver({}))

    with pytest.raises(TypeError, match="Received unsupported arguments: force"):
        provider.delete(sandbox_id="sbx-delete", force=True)
