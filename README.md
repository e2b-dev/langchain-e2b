# langchain-e2b

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-e2b?label=%20)](https://pypi.org/project/langchain-e2b/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-e2b)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-e2b)](https://pypistats.org/packages/langchain-e2b)

## Quick Install

```bash
pip install langchain-e2b
```

## Deep Agents SDK

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

## Deep Agents Code

Install `langchain-e2b` into the `dcode` environment, then run with the E2B
sandbox provider:

```bash
dcode --install langchain-e2b --package
export E2B_API_KEY=...
dcode --sandbox e2b
```

## What is this?

`langchain-e2b` adapts an existing E2B sandbox to the Deep Agents sandbox
protocol. It uses the low-level `e2b` SDK so Deep Agents can run shell commands
and move files through the standard Deep Agents sandbox interface.

For SDK use, this package intentionally does not hide E2B sandbox lifecycle
management. Use the E2B SDK to create, connect to, configure, and kill
sandboxes, then pass the connected sandbox object to `E2BSandbox`.

For Deep Agents Code, the package also exposes a `dcode` sandbox provider entry
point. The provider creates, reconnects to, and deletes E2B sandboxes for
`dcode --sandbox e2b`.

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
