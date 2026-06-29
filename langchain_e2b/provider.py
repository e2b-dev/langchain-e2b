"""E2B sandbox lifecycle provider for Deep Agents Code."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import NoReturn

import e2b
from deepagents_code.integrations.sandbox_provider import (
    SandboxInstallHint,
    SandboxNotFoundError,
    SandboxProvider,
    SandboxProviderMetadata,
)

from langchain_e2b.sandbox import DEFAULT_WORKDIR, AsyncE2BSandbox, E2BSandbox

DEFAULT_SANDBOX_TIMEOUT = 30 * 60
DEFAULT_COMMAND_TIMEOUT = 30 * 60

EnvResolver = Callable[[str], str | None]


def _default_resolve_env_var(name: str) -> str | None:
    if not name.startswith("DEEPAGENTS_CODE_"):
        prefixed = f"DEEPAGENTS_CODE_{name}"
        if prefixed in os.environ:
            return os.environ[prefixed] or None
    return os.environ.get(name) or None


def _resolve_optional_env_var(
    resolve_env_var: EnvResolver,
    name: str,
) -> str | None:
    value = resolve_env_var(name)
    return value or None


def _resolve_int(
    value: int | str | None,
    *,
    name: str,
    default: int,
    allow_zero: bool = False,
) -> int:
    if value is None:
        return default

    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be an integer number of seconds"
        raise ValueError(msg) from exc

    lower_bound = 0 if allow_zero else 1
    if resolved < lower_bound:
        requirement = "non-negative" if allow_zero else "positive"
        msg = f"{name} must be {requirement}"
        raise ValueError(msg)
    return resolved


def _raise_unsupported_kwargs(kwargs: dict[str, object]) -> NoReturn:
    names = ", ".join(sorted(kwargs))
    msg = f"Received unsupported arguments: {names}"
    raise TypeError(msg)


class E2BProvider(SandboxProvider):
    """Manage E2B sandboxes for Deep Agents Code."""

    _metadata = SandboxProviderMetadata(
        name="e2b",
        working_dir=DEFAULT_WORKDIR,
        install=SandboxInstallHint(kind="package", name="langchain-e2b"),
        supports_sandbox_id=True,
        supports_snapshot_name=False,
        backend_module="langchain_e2b",
    )

    def __init__(
        self,
        *,
        resolve_env_var: EnvResolver | None = None,
    ) -> None:
        """Initialize the provider without touching credentials.

        Deep Agents Code may instantiate providers only to read metadata during
        discovery, so credentials are resolved lazily by lifecycle methods.

        Args:
            resolve_env_var: Environment resolver used for provider settings.
        """
        self._resolve_env_var = resolve_env_var or _default_resolve_env_var

    @property
    def metadata(self) -> SandboxProviderMetadata:
        """Return metadata used by Deep Agents Code provider discovery."""
        return self._metadata

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int | str | None = None,
        template: str | None = None,
        workdir: str = DEFAULT_WORKDIR,
        command_timeout: int | str | None = None,
        **kwargs: object,
    ) -> E2BSandbox:
        """Get or create an E2B sandbox backend.

        Args:
            sandbox_id: Existing E2B sandbox ID, or `None` to create one.
            timeout: E2B sandbox lifetime in seconds.
            template: E2B template for new sandboxes.
            workdir: Working directory used by command execution.
            command_timeout: Default command timeout for the backend.
            **kwargs: Unsupported provider options.

        Returns:
            E2B sandbox backend.

        Raises:
            SandboxNotFoundError: If `sandbox_id` does not exist.
            TypeError: If unsupported provider options are passed.
            ValueError: If credentials or timeout settings are invalid.
        """
        if kwargs:
            _raise_unsupported_kwargs(kwargs)

        api_key = self._resolve_api_key()
        sandbox_timeout = self._resolve_sandbox_timeout(timeout)
        default_command_timeout = _resolve_int(
            command_timeout,
            name="command_timeout",
            default=DEFAULT_COMMAND_TIMEOUT,
            allow_zero=True,
        )

        try:
            if sandbox_id is not None:
                sandbox = e2b.Sandbox.connect(
                    sandbox_id,
                    timeout=sandbox_timeout,
                    api_key=api_key,
                )
            else:
                resolved_template = template or _resolve_optional_env_var(
                    self._resolve_env_var,
                    "E2B_TEMPLATE",
                )
                if resolved_template:
                    sandbox = e2b.Sandbox.create(
                        template=resolved_template,
                        timeout=sandbox_timeout,
                        api_key=api_key,
                    )
                else:
                    sandbox = e2b.Sandbox.create(
                        timeout=sandbox_timeout,
                        api_key=api_key,
                    )
        except e2b.SandboxNotFoundException as exc:
            if sandbox_id is None:
                raise
            raise SandboxNotFoundError(sandbox_id) from exc

        return E2BSandbox(
            sandbox=sandbox,
            workdir=workdir,
            timeout=default_command_timeout,
        )

    def delete(self, *, sandbox_id: str, **kwargs: object) -> None:
        """Kill an E2B sandbox.

        Args:
            sandbox_id: E2B sandbox ID.
            **kwargs: Unsupported provider options.

        Raises:
            SandboxNotFoundError: If `sandbox_id` does not exist.
            TypeError: If unsupported provider options are passed.
            ValueError: If credentials are missing.
        """
        if kwargs:
            _raise_unsupported_kwargs(kwargs)
        try:
            e2b.Sandbox.kill(sandbox_id, api_key=self._resolve_api_key())
        except e2b.SandboxNotFoundException as exc:
            raise SandboxNotFoundError(sandbox_id) from exc

    async def aget_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int | str | None = None,  # noqa: ASYNC109
        template: str | None = None,
        workdir: str = DEFAULT_WORKDIR,
        command_timeout: int | str | None = None,
        **kwargs: object,
    ) -> AsyncE2BSandbox:
        """Get or create an async E2B sandbox backend.

        Args:
            sandbox_id: Existing E2B sandbox ID, or `None` to create one.
            timeout: E2B sandbox lifetime in seconds.
            template: E2B template for new sandboxes.
            workdir: Working directory used by command execution.
            command_timeout: Default command timeout for the backend.
            **kwargs: Unsupported provider options.

        Returns:
            Async E2B sandbox backend.

        Raises:
            SandboxNotFoundError: If `sandbox_id` does not exist.
            TypeError: If unsupported provider options are passed.
            ValueError: If credentials or timeout settings are invalid.
        """
        if kwargs:
            _raise_unsupported_kwargs(kwargs)

        api_key = self._resolve_api_key()
        sandbox_timeout = self._resolve_sandbox_timeout(timeout)
        default_command_timeout = _resolve_int(
            command_timeout,
            name="command_timeout",
            default=DEFAULT_COMMAND_TIMEOUT,
            allow_zero=True,
        )

        try:
            if sandbox_id is not None:
                sandbox = await e2b.AsyncSandbox.connect(
                    sandbox_id,
                    timeout=sandbox_timeout,
                    api_key=api_key,
                )
            else:
                resolved_template = template or _resolve_optional_env_var(
                    self._resolve_env_var,
                    "E2B_TEMPLATE",
                )
                if resolved_template:
                    sandbox = await e2b.AsyncSandbox.create(
                        template=resolved_template,
                        timeout=sandbox_timeout,
                        api_key=api_key,
                    )
                else:
                    sandbox = await e2b.AsyncSandbox.create(
                        timeout=sandbox_timeout,
                        api_key=api_key,
                    )
        except e2b.SandboxNotFoundException as exc:
            if sandbox_id is None:
                raise
            raise SandboxNotFoundError(sandbox_id) from exc

        return AsyncE2BSandbox(
            sandbox=sandbox,
            workdir=workdir,
            timeout=default_command_timeout,
        )

    async def adelete(self, *, sandbox_id: str, **kwargs: object) -> None:
        """Kill an E2B sandbox asynchronously.

        Args:
            sandbox_id: E2B sandbox ID.
            **kwargs: Unsupported provider options.

        Raises:
            SandboxNotFoundError: If `sandbox_id` does not exist.
            TypeError: If unsupported provider options are passed.
            ValueError: If credentials are missing.
        """
        if kwargs:
            _raise_unsupported_kwargs(kwargs)
        try:
            await e2b.AsyncSandbox.kill(sandbox_id, api_key=self._resolve_api_key())
        except e2b.SandboxNotFoundException as exc:
            raise SandboxNotFoundError(sandbox_id) from exc

    def _resolve_api_key(self) -> str:
        api_key = _resolve_optional_env_var(self._resolve_env_var, "E2B_API_KEY")
        if not api_key:
            msg = (
                "No E2B API key found. Set E2B_API_KEY or DEEPAGENTS_CODE_E2B_API_KEY."
            )
            raise ValueError(msg)
        return api_key

    def _resolve_sandbox_timeout(self, timeout: int | str | None) -> int:
        env_timeout = (
            None
            if timeout is not None
            else _resolve_optional_env_var(self._resolve_env_var, "E2B_SANDBOX_TIMEOUT")
        )
        return _resolve_int(
            timeout if timeout is not None else env_timeout,
            name="timeout",
            default=DEFAULT_SANDBOX_TIMEOUT,
        )
