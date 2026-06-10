# langchain-e2b

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-e2b?label=%20)](https://pypi.org/project/langchain-e2b/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-e2b)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-e2b)](https://pypistats.org/packages/langchain-e2b)

## Quick Install

```bash
pip install langchain-e2b
```

```python
from e2b import Sandbox
from langchain_e2b import E2BSandbox

e2b_sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=e2b_sandbox)

try:
    result = backend.execute("echo hello")
    print(result.output)
finally:
    e2b_sandbox.kill()
```

## What is this?

`langchain-e2b` adapts an existing E2B sandbox to the Deep Agents sandbox
protocol. It uses the low-level `e2b` SDK so Deep Agents can run shell commands
and move files through the standard Deep Agents sandbox interface.

This package intentionally does not hide E2B sandbox lifecycle management. Use
the E2B SDK to create, connect to, configure, and kill sandboxes, then pass the
connected sandbox object to `E2BSandbox`.

## Contributing

Contributions are welcome. Keep the adapter focused on implementing the Deep
Agents sandbox backend protocol over the official E2B SDK.

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
