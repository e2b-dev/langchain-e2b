# langchain-e2b

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-e2b?label=%20)](https://pypi.org/project/langchain-e2b/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-e2b)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-e2b)](https://pypistats.org/packages/langchain-e2b)

## Quick Install

```bash
pip install langchain-e2b
```

```python
from langchain_e2b import E2BProvider

provider = E2BProvider()
sandbox = provider.get_or_create()

try:
    result = sandbox.execute("echo hello")
    print(result.output)
finally:
    provider.delete(sandbox_id=sandbox.id)
```

## What is this?

`langchain-e2b` adapts E2B sandboxes to the Deep Agents sandbox protocol. It
uses the low-level `e2b` SDK so Deep Agents can create or reconnect to
sandboxes, run shell commands, and move files through the standard Deep Agents
sandbox interface.

Use `E2BProvider` when you want the package to manage sandbox lifecycle. Use
`E2BSandbox` directly when you already have an E2B SDK sandbox object.

## Contributing

Contributions are welcome. Keep the adapter focused on implementing the Deep
Agents sandbox protocol over the official E2B SDK.

## Development

```bash
uv sync --group test
make test
make lint
```

Integration tests require `E2B_API_KEY`:

```bash
E2B_API_KEY=... make integration_tests
```
