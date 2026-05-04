# Agent Instructions

## Code Generation Rules

### Language for Code Comments & Docstrings

Always generate comments, docstrings, and any text embedded in code in **English**,
regardless of the language the user communicates in.

Docstrings should follow the **Google style** format:

```python
def function_name(param1, param2):
    """Short one-line description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: When this exception is raised.
    ```

### Test Execution
Run tests using the project's task runner:
- **Unit tests:** `uv run poe test`
- **Coverage:** `uv run poe test_cov`

Never use `pytest` directly - always go through `poe` via `uv run`.

`poe test` accepts the same arguments as `pytest`, so you can pass directories, files,
or individual tests:
- `uv run poe tests/tests/` - run all tests in a directory
- `uv run poe tests/test_file.py` - run a single test file
- `uv run poe tests/test_file.py::test_function` - run a single test

### Typography
Always use regular hyphens (`-`) instead of em-dashes or en-dashes. Avoid custom
characters such as emojis or decorative symbols.

## Project Structure

- **`src/`** — Main source code directory containing the `skrough` package.
- **`tests/`** — Test suite for the project.
- **`examples/`** — Jupyter notebooks used for documentation.
- **`docs/`** — Main documentation directory (JupyterBook). Notebooks from `examples/`
  are linked/included here.

### Directories to Ignore
During normal work (unless explicitly asked otherwise or during refactoring), do **not**
pay attention to the following directories:

- **`trash/`** — Contains discarded/obsolete files.
- **`dev/`** — Contains work-in-progress files that may be outdated.

These directories are not actively maintained and should be left as-is.
