FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

COPY "predict.py" "predict_app.py" "xgb_pipeline.pkl" ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict_app:app", "--host", "0.0.0.0", "--port", "9696"]