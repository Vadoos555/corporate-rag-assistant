
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

COPY data/pdfs ./data/pdfs

EXPOSE 8501

CMD ["streamlit", "run", "src/corporate_rag/app.py", "--server.address=0.0.0.0", "--server.port=8501"]