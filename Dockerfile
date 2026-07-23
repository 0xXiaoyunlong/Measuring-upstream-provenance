# Reproduce the canonical run in a pinned, single-threaded container.
#   docker build -t upstream-provenance-canonical .
#   docker run --rm upstream-provenance-canonical
#
# Note: the three-decimal result (0.703) is expected to reproduce; the
# full-precision macro AUROC can differ by a hundredth on a different BLAS/CPU
# because a few single-event folds are near ranking ties (see README).
FROM python:3.11.9-slim

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements-exact.txt .
RUN pip install --no-cache-dir -r requirements-exact.txt

COPY . .

CMD ["python", "run_all.py"]
