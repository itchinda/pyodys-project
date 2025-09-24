from abc import ABC, abstractmethod
import numpy as np

class Scheme(ABC):
    """Abstract base class for time integration schemes."""
    def __init__(self, order: int = None):
        if order is None:
            raise ValueError("Scheme order must be specified")
        self.order = order

    @property
    @abstractmethod
    def n_stages(self) -> int:
        """Return number of stages (for RK) or number of past steps (for BDF)."""
        raise NotImplementedError("Subclasses must implement n_stages property.")

