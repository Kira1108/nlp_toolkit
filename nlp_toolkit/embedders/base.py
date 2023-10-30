import numpy as np
from typing import Protocol

class BaseEmbedder(Protocol):
    def __call__(self, texts) -> np.ndarray:
        ...