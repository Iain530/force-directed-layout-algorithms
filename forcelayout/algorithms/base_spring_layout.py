from .node import Node
from ..utils import jiggle, get_size, mean
from ..distance import euclidean
from itertools import combinations
from typing import Callable, Tuple, List, Dict, FrozenSet
from abc import ABC, abstractmethod
import numpy as np
import math


class BaseSpringLayout(ABC):
    """
    Base class for a spring layout algorithm
    """
    def __init__(self, dataset: np.ndarray = None, nodes: List[Node] = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean,
                 iterations: int = 50, target_node_speed: float = 0.0,
                 enable_cache: bool = True) -> None:
        assert iterations >= 0, "iterations must be non-negative"
        assert dataset is not None or nodes is not None, "must provide either dataset or nodes"

        self.nodes: List[Node] = nodes if nodes is not None else self._build_nodes(dataset)
        self.iterations: int = iterations
        self.data_size_factor: float = 1
        self._i: int = 0  # current iteration
        self.distance_fn: Callable[[np.ndarray, np.ndarray], float] = distance_fn
        self._average_speeds: List[float] = list()
        self.target_node_speed: float = target_node_speed
        self.enable_cache: bool = enable_cache
        if enable_cache:
            self.distances: Dict[FrozenSet[Node], float] = dict()
        else:
            # Change the distance function
            self.distance = self.distance_no_cache

    def current_iteration(self) -> int:
        return self._i

    def get_positions(self) -> np.ndarray:
        return np.array([(n.x, n.y) for n in self.nodes])

    def set_positions(self, positions: np.ndarray) -> None:
        for pos, node in zip(positions, self.nodes):
            node.x, node.y = pos

    def get_stress(self) -> float:
        distance_diff: float = 0.0
        actual_distance: float = 0.0
        for source, target in combinations(self.nodes, 2):
            high_d_distance = self.distance(source, target, cache=False)
            low_d_distance = math.sqrt((target.x - source.x)**2 + (target.y - source.y)**2)
            distance_diff += (high_d_distance - low_d_distance)**2
            actual_distance += low_d_distance**2
        if actual_distance == 0:
            return math.inf
        return math.sqrt(distance_diff / actual_distance)

    def get_memory(self) -> int:
        return get_size(self)

    def average_speed(self) -> float:
        """ Return the 5-running mean of the average node speeds """
        return mean(self._average_speeds[-5:]) if len(self._average_speeds) > 0 else np.inf

    def spring_layout(self, alpha: float = 1, return_after: int = None) -> np.ndarray:
        """
        Method to perform the main spring layout calculation, move the nodes self.iterations
        number of times unless return_after is given.
        If return_after is specified then the nodes will be moved the value of return_after
        times. Subsequent calls to spring_layout will continue from the previous number of
        iterations.
        """
        if return_after is not None and self.average_speed() > self.target_node_speed:
            for i in range(return_after):
                self._spring_layout(alpha=alpha)
                self._i += 1
        else:
            while self.average_speed() > self.target_node_speed and self._i < self.iterations:
                self._spring_layout(alpha=alpha)
                self._i += 1
        # Return calculated positions for datapoints
        return self.get_positions()

    def distance_no_cache(self, source: Node, target: Node, cache: bool = False) -> float:
        """ Distance function to use when self.disable_cache = True """
        return self.distance_fn(source.datapoint, target.datapoint)

    def distance(self, source: Node, target: Node, cache: bool = False) -> float:
        """
        Returns the high dimensional distance between two nodes at source and target
        index using self.distance_fn
        """
        pair = frozenset({source, target})
        if pair in self.distances:
            return self.distances[pair]
        distance = self.distance_fn(source.datapoint, target.datapoint)
        if cache:
            self.distances[pair] = distance
        return distance

    @abstractmethod
    def _spring_layout(self, alpha: float) -> None:
        """
        Perform one iteration of the spring layout
        """
        pass

    def _force(self, current_distance, real_distance, alpha: float=1) -> float:
        return (current_distance - real_distance) * alpha * self.data_size_factor / current_distance

    def _calculate_velocity(self, source: Node, target: Node, alpha: float=1,
                            cache_distance: bool = False) -> Tuple[float, float]:
        """
        Calculate the spring force to apply between two nodes i and j
        """
        x, y = self._current_distance(source, target)
        dist = math.hypot(x, y)
        real_dist = self.distance(source, target, cache=cache_distance)
        force = self._force(dist, real_dist, alpha)
        return (x * force, y * force)

    def _current_distance(self, source: Node, target: Node) -> Tuple[float, float]:
        """
        Calculate the current 2d layout distance between two nodes.
        Apply a small non zero random value to remove values of zero
        """
        x = target.x - source.x
        y = target.y - source.y
        # x and y must be non zero
        x = x if x else jiggle()
        y = y if y else jiggle()
        return x, y

    def _set_velocity(self, source: Node, target: Node, alpha: float,
                      cache_distance: bool = False) -> None:
        """
        Calculate the force between two nodes and update the
        velocity of both nodes
        """
        vx, vy = self._calculate_velocity(source, target, alpha=alpha,
                                          cache_distance=cache_distance)
        source.add_velocity(vx, vy)
        target.add_velocity(-vx, -vy)

    def _apply_velocities(self) -> None:
        """
        Apply the current velocity of each node to its position
        and reset velocity
        """
        total: float = 0.0
        for node in self.nodes:
            total += math.hypot(node.vx, node.vy)
            node.apply_velocity()
        total /= len(self.nodes)
        self._average_speeds.append(total)

    def _build_nodes(self, dataset: np.ndarray) -> List[Node]:
        """
        Contrust a Node for each datapoint
        """
        return list(np.apply_along_axis(Node, axis=1, arr=dataset))
