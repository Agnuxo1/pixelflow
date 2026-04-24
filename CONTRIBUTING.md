# Contributing to pixelflow

Thanks for your interest. This is a small research library; contributions are
welcome but the bar is simple: **honest code, tested behaviour, no hype**.

## Ground rules

1. **No fake benchmarks.** Report what actually ran. Failures are fine to
   publish; fictional numbers are not.
2. **Tests before features.** A new rule, backend, or task must land with tests
   that pass on CI (CPU-only is fine — GPU tests skip when unavailable).
3. **Keep the API contract.** Changes to public types in
   `docs/API_CONTRACT.md` require an issue + discussion first.
4. **No emojis** in code, comments, commits, or PR descriptions. Keep tone
   professional.

## Dev setup

```bash
git clone https://github.com/Agnuxo1/pixelflow.git
cd pixelflow
pip install -e ".[all]"
pytest tests/ -v
```

## Adding a new CA rule

1. Add a numpy `step` function to `pixelflow/core/rules.py`.
2. Add the equivalent GLSL 3.30 fragment shader as a string in the same file.
3. Add unit test: `tests/test_rules.py` — numerical agreement CPU vs GLSL on a
   fixture, stability under many steps.
4. Add example usage in `examples/` if the rule enables a new task.

## Adding a backend

1. New file: `pixelflow/backends/<name>.py` implementing `run_<name>(...)`.
2. Register it in `pixelflow/backends/__init__.py::get_backend`.
3. Parity test: `tests/test_<name>_backend.py` comparing against CPU reference.

## PR checklist

- [ ] Tests added and passing locally
- [ ] `ruff check` clean
- [ ] No print statements in library code
- [ ] Type hints on public functions
- [ ] README / docs updated if public API changed

## License

By submitting a PR, you agree your contribution is licensed under Apache-2.0.
