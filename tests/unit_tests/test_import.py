from __future__ import annotations

import langchain_e2b


def test_import_e2b() -> None:
    assert langchain_e2b is not None


def test_public_exports_only_sandbox() -> None:
    assert langchain_e2b.__all__ == ["E2BSandbox"]
    assert hasattr(langchain_e2b, "E2BSandbox")
    assert not hasattr(langchain_e2b, "E2BProvider")
