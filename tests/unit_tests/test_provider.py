from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import e2b
import pytest

from langchain_e2b import E2BProvider, E2BSandbox
from langchain_e2b.provider import DEFAULT_SANDBOX_TIMEOUT

if TYPE_CHECKING:
    from collections.abc import Callable


def _resolver(values: dict[str, str]) -> Callable[[str], str | None]:
    return values.get


def _sandbox(sandbox_id: str = "sbx-test") -> e2b.Sandbox:
    return cast("e2b.Sandbox", MagicMock(sandbox_id=sandbox_id))


def test_provider_resolves_api_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider can read `E2B_API_KEY` without an explicit constructor key."""
    monkeypatch.setenv("E2B_API_KEY", "env-key")
    sandbox = _sandbox()

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider()
        backend = provider.get_or_create()

    create.assert_called_once_with(
        timeout=DEFAULT_SANDBOX_TIMEOUT,
        api_key="env-key",
    )
    assert isinstance(backend, E2BSandbox)
    assert backend.id == "sbx-test"


def test_provider_requires_api_key() -> None:
    """A missing API key should fail before any SDK call is attempted."""
    with pytest.raises(ValueError, match="No E2B API key found"):
        E2BProvider(resolve_env_var=_resolver({}))


def test_provider_creates_sandbox_with_default_options() -> None:
    """Creating a sandbox uses the package-owned lifecycle implementation."""
    sandbox = _sandbox("sbx-new")

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider(api_key="fake-key", resolve_env_var=_resolver({}))
        backend = provider.get_or_create(timeout=2)

    create.assert_called_once_with(
        timeout=DEFAULT_SANDBOX_TIMEOUT,
        api_key="fake-key",
    )
    assert backend.id == "sbx-new"


def test_provider_creates_sandbox_with_template_and_timeout() -> None:
    """Template and sandbox lifetime are configured by E2B environment values."""
    sandbox = _sandbox("sbx-template")
    resolve_env_var = _resolver(
        {
            "E2B_TEMPLATE": "custom-template",
            "E2B_SANDBOX_TIMEOUT": "7200",
        }
    )

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.create",
        return_value=sandbox,
    ) as create:
        provider = E2BProvider(api_key="fake-key", resolve_env_var=resolve_env_var)
        backend = provider.get_or_create()

    create.assert_called_once_with(
        template="custom-template",
        timeout=7200,
        api_key="fake-key",
    )
    assert backend.id == "sbx-template"


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
def test_provider_rejects_invalid_sandbox_timeout(value: str) -> None:
    """Timeout configuration should fail with a clear error."""
    resolve_env_var = _resolver({"E2B_SANDBOX_TIMEOUT": value})

    with pytest.raises(ValueError, match="E2B_SANDBOX_TIMEOUT"):
        E2BProvider(api_key="fake-key", resolve_env_var=resolve_env_var)


def test_provider_connects_existing_sandbox() -> None:
    """Providing `sandbox_id` reconnects instead of creating a new sandbox."""
    sandbox = _sandbox("sbx-existing")

    with patch(
        "langchain_e2b.provider.e2b.Sandbox.connect",
        return_value=sandbox,
    ) as connect:
        provider = E2BProvider(api_key="fake-key", resolve_env_var=_resolver({}))
        backend = provider.get_or_create(sandbox_id="sbx-existing")

    connect.assert_called_once_with(
        "sbx-existing",
        timeout=DEFAULT_SANDBOX_TIMEOUT,
        api_key="fake-key",
    )
    assert backend.id == "sbx-existing"


def test_provider_translates_missing_sandbox_to_key_error() -> None:
    """Deep Agents adapters can map missing sandboxes without importing E2B SDK."""
    with patch(
        "langchain_e2b.provider.e2b.Sandbox.connect",
        side_effect=e2b.SandboxNotFoundException("missing"),
    ):
        provider = E2BProvider(api_key="fake-key", resolve_env_var=_resolver({}))

        with pytest.raises(KeyError) as exc_info:
            provider.get_or_create(sandbox_id="sbx-missing")

    assert exc_info.value.args == ("sbx-missing",)


def test_provider_rejects_unsupported_get_options() -> None:
    """Unsupported lifecycle options fail in the package provider."""
    provider = E2BProvider(api_key="fake-key", resolve_env_var=_resolver({}))

    with pytest.raises(TypeError, match="Received unsupported arguments: metadata"):
        provider.get_or_create(metadata={"purpose": "test"})


def test_provider_deletes_sandbox() -> None:
    """Delete delegates to the E2B SDK lifecycle operation."""
    with patch("langchain_e2b.provider.e2b.Sandbox.kill") as kill:
        provider = E2BProvider(api_key="fake-key", resolve_env_var=_resolver({}))
        provider.delete(sandbox_id="sbx-delete")

    kill.assert_called_once_with("sbx-delete", api_key="fake-key")


def test_provider_rejects_unsupported_delete_options() -> None:
    """Delete only accepts the sandbox ID."""
    provider = E2BProvider(api_key="fake-key", resolve_env_var=_resolver({}))

    with pytest.raises(TypeError, match="Received unsupported arguments: force"):
        provider.delete(sandbox_id="sbx-delete", force=True)
