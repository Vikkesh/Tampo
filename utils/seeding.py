"""
Global seeding for reproducible training and benchmarking runs.

Training in this codebase was previously unseeded, so two Colab sessions started
from scratch on identical code and config produced policies that differed enough
to change reported makespan/energy by orders of magnitude.  Call `set_seed()`
once, before the environment and networks are constructed, in every entry point.

Reproducibility caveat: a fixed seed makes a *single* run repeatable.  It does not
make a single run *representative*.  Report mean +/- std over several seeds
(see `SEEDS`) before drawing any conclusion about one encoder beating another.
"""

import os
import random

import numpy as np
import torch

# Suggested seed set for multi-seed benchmark reporting.
SEEDS = (0, 1, 2, 3, 4)


def set_seed(seed: int, deterministic_torch: bool = False) -> int:
    """
    Seed every RNG this project draws from.

    Args:
        seed: The seed value.
        deterministic_torch: If True, force deterministic cuDNN/cuBLAS kernels.
            Costs throughput and makes some ops raise instead of silently using a
            nondeterministic path.  Leave False for normal training; enable it when
            bit-for-bit reproduction of a specific run matters.

    Returns:
        The seed, so callers can log it alongside their results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PYTHONHASHSEED only takes effect for interpreters started after it is set,
    # so this is a no-op for the current process's str/bytes hashing.  It is set
    # anyway so that subprocesses (and Colab cell re-execs) inherit it.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if deterministic_torch:
        # cuBLAS needs this to make matmul reductions deterministic on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
