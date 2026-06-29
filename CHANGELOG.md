<!-- markdownlint-disable MD024 -->

# Changelog

## 0.0.5

- Added `AsyncE2BSandbox` backed by `e2b.AsyncSandbox`.
- Added async E2B provider lifecycle methods for `dcode` async flows.

## 0.0.4

- Added a Deep Agents Code sandbox provider entry point so `dcode --sandbox e2b`
  works after installing `langchain-e2b` into the `dcode` environment.

## 0.0.3

- Removed the `E2BProvider` lifecycle helper. Use the E2B SDK to create,
  configure, connect to, and kill sandboxes, then wrap them with `E2BSandbox`.

## 0.0.2

- Added the `E2BProvider` lifecycle helper.

## 0.0.1

- Initial E2B sandbox integration for Deep Agents.
