import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def rng():
    return random.Random(1234)


#: Input shapes every sort must handle. Edge cases first - empty, single,
#: all-equal and already-sorted inputs break more implementations than large
#: random ones do.
EDGE_CASES = [
    [],
    [1],
    [2, 1],
    [1, 1, 1, 1],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
    [0, 0, 1, 0, 1, 1, 0],
]
