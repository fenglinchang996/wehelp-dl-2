# WeHelp Assignment

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Run from the project directory:

```bash
uv run fastapi dev app.py
# or
uv run fastapi dev
```

Open <http://127.0.0.1:8000/>. The first launch downloads the CKIP models.

Without uv, install the dependencies listed in `pyproject.toml` using Python 3.12, then run `fastapi dev app.py`.
