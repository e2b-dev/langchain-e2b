"""E2B sandbox integration for Deep Agents."""

from e2b import ConnectionConfig as _ConnectionConfig

from langchain_e2b._version import __version__ as _package_version
from langchain_e2b.sandbox import AsyncE2BSandbox, E2BSandbox

# Tag E2B SDK traffic before callers can construct a connection configuration.
_ConnectionConfig.set_integration(f"langchain-e2b/{_package_version}")

__all__ = ["AsyncE2BSandbox", "E2BSandbox"]
