"""
Template estimation without running out of /dev/shm.

SpikeInterface's estimate_templates_with_accumulator (used when a SortingAnalyzer is created, to
find each unit's main channel, and when templates are computed without stored waveforms, as in the
auto-merge step) puts one accumulator per worker in POSIX shared memory: n_jobs x units x samples x
channels x float32, twice when it also computes the standard deviation. Docker gives a container
64 MB of /dev/shm by default, and AWS Lambda (or `docker run --ipc none`) has none, so a large
sorting fails with SIGBUS or FileNotFoundError('/psm_...').

template_jobs() picks how many workers fit in the free /dev/shm (at most THREADS). When not even
one fits, or there is no /dev/shm, local_shared_memory() swaps SpikeInterface's shared arrays for
ordinary in-process arrays for the duration of the call, with a single worker (n_jobs=1 runs in the
main process, so nothing has to be shared).
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import uuid

import numpy as np

log = logging.getLogger("ephys-curate")

SHM_PATH = "/dev/shm"
SAFETY = 0.8          # use at most 80 % of the free space


def shm_free_bytes(path: str = SHM_PATH) -> int | None:
    try:
        st = os.statvfs(path)
    except (OSError, AttributeError):
        return None
    return int(st.f_bavail * st.f_frsize)


@functools.lru_cache(maxsize=None)
def shm_usable(path: str = SHM_PATH) -> bool:
    """Whether POSIX shared memory works here (multiprocessing locks and SharedMemory need it).

    Cached: the first call (at startup, before any patching) decides for the run."""
    try:
        from multiprocessing.shared_memory import SharedMemory

        shm = SharedMemory(create=True, size=16)
        shm.close()
        shm.unlink()
        return True
    except (OSError, ValueError):
        return False


def accumulator_bytes_per_worker(n_units: int, n_channels: int, fs: float, ms: float = 5.0) -> int:
    """Upper bound for one worker: units x samples (ms) x channels x float32 x 2 (mean and std)."""
    n_samples = int(np.ceil(fs * ms / 1000.0)) + 1
    return int(n_units) * n_samples * int(n_channels) * 4 * 2


def max_sparse_channels(positions: np.ndarray, radius_um: float) -> int:
    """The largest number of channels within radius_um of one channel (the sparse width)."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or len(pos) == 0:
        return 1
    best = 1
    for i in range(len(pos)):
        best = max(best, int((np.sqrt(((pos - pos[i]) ** 2).sum(1)) <= radius_um).sum()))
    return best


def template_jobs(n_units: int, recording, radius_um: float, threads: int, path: str = SHM_PATH) -> tuple[int, bool, str]:
    """(n_jobs, use_local_memory, reason) for SpikeInterface's template accumulator."""
    free = shm_free_bytes(path)
    n_ch = max_sparse_channels(recording.get_channel_locations(), radius_um)
    # main-channel estimation runs before the sparsity is known: bound it by the dense channel count
    per = accumulator_bytes_per_worker(n_units, recording.get_num_channels(), recording.sampling_frequency)
    per_sparse = accumulator_bytes_per_worker(n_units, n_ch, recording.sampling_frequency)
    if free is None:
        return 1, True, f"no {path}: template estimation in process memory, 1 job"
    fits = int(free * SAFETY // per) if per else threads
    if fits >= 1:
        n = max(1, min(threads, fits))
        return n, False, (f"{path} has {free / 2**20:.0f} MB free, {per / 2**20:.1f} MB per template worker: "
                          f"{n} job(s)")
    fits_sparse = int(free * SAFETY // per_sparse) if per_sparse else 0
    return 1, True, (f"{path} has {free / 2**20:.0f} MB free, {per / 2**20:.1f} MB needed per template worker "
                     f"({per_sparse / 2**20:.1f} MB sparse{'' if fits_sparse else ', also too much'}): "
                     "template estimation in process memory, 1 job")


class _LocalShm:
    """A stand-in for multiprocessing.shared_memory.SharedMemory backed by process memory."""

    _registry: dict[str, memoryview] = {}

    def __init__(self, name=None, create=False, size=0, **_):
        if create:
            self.name = name or f"local_{uuid.uuid4().hex[:12]}"
            self.buf = memoryview(bytearray(max(int(size), 1)))
            _LocalShm._registry[self.name] = self.buf
        else:
            if name not in _LocalShm._registry:
                raise FileNotFoundError(name)
            self.name, self.buf = name, _LocalShm._registry[name]
        self.size = len(self.buf)

    def close(self):
        pass

    def unlink(self):
        _LocalShm._registry.pop(self.name, None)


def _local_make_shared_array(shape, dtype):
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = _LocalShm(create=True, size=nbytes)
    arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    arr[:] = 0
    return arr, shm


@contextlib.contextmanager
def local_shared_memory(enabled: bool = True):
    """Within the block, SpikeInterface's shared arrays live in process memory.

    Every loaded spikeinterface module's make_shared_array / SharedMemory, and
    multiprocessing.shared_memory.SharedMemory, are swapped for in-process versions. Only valid
    while nothing runs in another process: n_jobs=1, or the thread pool engine.
    """
    if not enabled:
        yield
        return
    import multiprocessing.shared_memory as msm
    import sys

    import spikeinterface.core.waveform_tools  # noqa: F401 - make sure the main users are loaded

    saved = [(msm, "SharedMemory", msm.SharedMemory)]
    msm.SharedMemory = _LocalShm
    for name, mod in list(sys.modules.items()):
        if not name.startswith("spikeinterface") or mod is None:
            continue
        for attr, repl in (("make_shared_array", _local_make_shared_array), ("SharedMemory", _LocalShm)):
            if hasattr(mod, attr):
                saved.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, repl)
    try:
        yield
    finally:
        for mod, attr, val in reversed(saved):
            setattr(mod, attr, val)
