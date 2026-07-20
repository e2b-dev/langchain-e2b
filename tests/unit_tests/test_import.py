from __future__ import annotations

from e2b import ConnectionConfig

import langchain_e2b
from langchain_e2b._version import __version__


def test_import_e2b() -> None:
    assert langchain_e2b is not None


def test_e2b_integration_attribution() -> None:
    config = ConnectionConfig()

    user_agent_products = config.headers["User-Agent"].split()
    assert f"langchain-e2b/{__version__}" in user_agent_products


def test_public_exports_sandboxes() -> None:
    assert langchain_e2b.__all__ == ["AsyncE2BSandbox", "E2BSandbox"]
    assert hasattr(langchain_e2b, "AsyncE2BSandbox")
    assert hasattr(langchain_e2b, "E2BSandbox")
    assert not hasattr(langchain_e2b, "E2BProvider")
