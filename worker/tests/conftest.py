"""Test-process environment guards.

torch and xgboost each bundle their own libomp; loading both in ONE process aborts on macOS
with OMP Error #15, and the documented escape hatch alone (KMP_DUPLICATE_LIB_OK) trades the
abort for a thread-pool deadlock (observed: pytest idle-hung at ~0% CPU). Pinning OpenMP to
one thread makes the duplicate runtimes coexist deterministically — measured with an
xgboost→torch→xgboost round-trip. Only the test suite ever mixes the two: every production
CLI loads at most one (train-stroke → torch, train-stroke-gbdt → xgboost), so both knobs are
scoped to tests, where workloads are tiny and single-threaded is fine.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
