# Real-time scoring + API node.   Build for IBM Z / LinuxONE:
#   docker buildx build --platform linux/s390x -t cashout-rt:s390x .
# (UNTESTED here: the authoring sandbox had no Docker. Build it once on your LPAR / buildx before the demo.)
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt
COPY src ./src
COPY api.py .
COPY data ./data
COPY models/hgb_portable.npz ./models/hgb_portable.npz
RUN useradd --system app && mkdir -p outputs/audit && chown -R app outputs
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
