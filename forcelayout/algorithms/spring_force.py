from .base_spring_layout import BaseSpringLayout
from forcelayout.distance import euclidean
from itertools import combinations
from typing import Callable
import numpy as np


class SpringForce(BaseSpringLayout):
    """
    Basic brute force spring layout implementation
    """
    def __init__(self, dataset: np.ndarray,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean,
                 iterations: int = 50, target_node_speed: float=0.0,
                 enable_cache: bool = False) -> None:
        super().__init__(dataset=dataset, distance_fn=distance_fn, iterations=iterations,
                         target_node_speed=target_node_speed, enable_cache=enable_cache)
        self.data_size_factor = 0.5 / (len(dataset) - 1)

    def _spring_layout(self, alpha: float = 1) -> None:
        # Calculate velcoities
        for source, target in combinations(self.nodes, 2):
            self._set_velocity(source, target, alpha)

        # Apply velocities
        self._apply_velocities()
