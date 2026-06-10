"""E2B sandbox lifecycle provider."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import NoReturn

import e2b

from langchain_e2b.sandbox import E2BSandbox

DEFAULT_SANDBOX_TIMEOUT = 30 * 60
EnvResolver = Callable[[str], str | None]


def _default_resolve_env_var(name: str) -> str | None:
    return os.environ.get(name)


def _resolve_sandbox_timeout(resolve_env_var: EnvResolver) -> int:
    value = resolve_env_var("E2B_SANDBOX_TIMEOUT")
    if not value:
        return DEFAULT_SANDBOX_TIMEOUT

    try:
        timeout = int(value)
    except ValueError as exc:
        msg = "E2B_SANDBOX_TIMEOUT must be an integer number of seconds"
        raise ValueError(msg) from exc

    if timeout <= 0:
        msg = "E2B_SANDBOX_TIMEOUT must be positive"
        raise ValueError(msg)
    return timeout


def _raise_unsupported_kwargs(kwargs: dict[str, object]) -> NoReturn:
    names = ", ".join(sorted(kwargs))
    msg = f"Received unsupported arguments: {names}"
    raise TypeError(msg)


class E2BProvider:
    """Manage E2B sandbox lifecycle for Deep Agents."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        resolve_env_var: EnvResolver | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: E2B API key. If omitted, `E2B_API_KEY` is resolved through
                `resolve_env_var`.
            resolve_env_var: Environment resolver used for provider settings.

        Raises:
            ValueError: If no API key is available or timeout configuration is
                invalid.
        """
        self._resolve_env_var = resolve_env_var or _default_resolve_env_var

        resolved_api_key = api_key or self._resolve_env_var("E2B_API_KEY")
        if not resolved_api_key:
            msg = "No E2B API key found. Set E2B_API_KEY."
            raise ValueError(msg)

        self._api_key = resolved_api_key
        self._template = self._resolve_env_var("E2B_TEMPLATE") or None
        self._sandbox_timeout = _resolve_sandbox_timeout(self._resolve_env_var)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: object,
    ) -> E2BSandbox:
        """Get or create an E2B sandbox backend.

        Args:
            sandbox_id: Existing E2B sandbox ID, or None to create a sandbox.
            timeout: Accepted for provider-interface compatibility. E2B sandbox
                lifetime is configured with `E2B_SANDBOX_TIMEOUT`.
            **kwargs: Unsupported provider options.

        Returns:
            E2B sandbox backend.

        Raises:
            KeyError: If `sandbox_id` does not exist.
            TypeError: If unsupported provider options are passed.
        """
        del timeout
        if kwargs:
            _raise_unsupported_kwargs(kwargs)

        if sandbox_id is not None:
            try:
                sandbox = e2b.Sandbox.connect(
                    sandbox_id,
                    timeout=self._sandbox_timeout,
                    api_key=self._api_key,
                )
            except e2b.SandboxNotFoundException as exc:
                raise KeyError(sandbox_id) from exc
        elif self._template:
            sandbox = e2b.Sandbox.create(
                template=self._template,
                timeout=self._sandbox_timeout,
                api_key=self._api_key,
            )
        else:
            sandbox = e2b.Sandbox.create(
                timeout=self._sandbox_timeout,
                api_key=self._api_key,
            )

        return E2BSandbox(sandbox=sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: object) -> None:
        """Kill an E2B sandbox.

        Args:
            sandbox_id: E2B sandbox ID.
            **kwargs: Unsupported provider options.

        Raises:
            TypeError: If unsupported provider options are passed.
        """
        if kwargs:
            _raise_unsupported_kwargs(kwargs)
        e2b.Sandbox.kill(sandbox_id, api_key=self._api_key)
