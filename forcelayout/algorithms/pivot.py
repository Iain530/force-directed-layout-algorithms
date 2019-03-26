from .hybrid import Hybrid
from .node import Node
from ..utils import point_on_circle, random_sample_set, flexible_intersection, show_progress
from ..distance import euclidean
from typing import Callable, List, Dict, Set
import numpy as np
import math


class Pivot(Hybrid):
    """
    An implementation of Chalmers and Morrison pivot layout extending the 2002 Hybrid layout.
    It performs the '96 neighbour sampling algorithm, on a sample set, sqrt(n) samples of the data.
    sqrt(sqrt(n)) pivots are chosen from the sample set and distances to samples are precomputed
    into buckets. Other data points are then interpolated into the model by querying the pivot
    buckets. Finally, '96 neighbour sampling is performed to clean up the layout.
    """

    def __init__(self, dataset: np.ndarray = None, nodes: List[Node] = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean,
                 sample_layout_iterations: int = 75, remainder_layout_iterations: int = 5,
                 refine_layout_iterations: int = 5, random_sample_size: int = 15,
                 preset_sample: List[int] = None, pivot_set_size: int = 10,
                 target_node_speed: float=0.0, enable_cache: bool=True) -> None:
        super().__init__(dataset=dataset, nodes=nodes, distance_fn=distance_fn,
                         sample_layout_iterations=sample_layout_iterations,
                         remainder_layout_iterations=remainder_layout_iterations,
                         refine_layout_iterations=refine_layout_iterations,
                         random_sample_size=random_sample_size,
                         preset_sample=preset_sample,
                         target_node_speed=target_node_speed,
                         enable_cache=enable_cache)
        self.pivot_set_size: int = min(self.sample_set_size, pivot_set_size)
        self.pivot_indexes: List[int] = random_sample_set(self.pivot_set_size, self.sample_set_size)
        self.pivots: List[Node] = [self.sample[i] for i in self.pivot_indexes]
        target_num_buckets: int = round(math.sqrt(self.sample_set_size))
        self.bucket_size: int = math.ceil(self.sample_set_size / target_num_buckets)
        self.num_buckets: int = math.ceil(self.sample_set_size / self.bucket_size)
        self.bucket_distances: Dict[Node, List[float]] = {p: [] for p in self.pivots}
        self.pivot_buckets: Dict[Node, List[Set[Node]]] = {
            p: [set() for i in range(self.num_buckets)] for p in self.pivots
        }

    def _remainder_layout_stage(self, alpha: float) -> None:
        """
        Layout the remainder nodes close to similar nodes from the
        sample set using pivot based searching
        """
        # Preprocess pivots for faster queries
        for pivot in self.pivots:
            self._preprocess_pivot(pivot)

        # Layout the remainder
        number_remainders = len(self.remainder_indexes)
        for i in range(number_remainders):
            show_progress(i, number_remainders, stage='Remainder')
            self._place_near_parent(self.remainder_indexes[i], alpha)

    def _preprocess_pivot(self, pivot: Node) -> None:
        sorted_sample = self.sample.copy()
        sorted_sample.sort(key=lambda t: self.distance(pivot, t))

        buckets = self.pivot_buckets[pivot]

        for i in range(len(sorted_sample)):
            bucket_index = i // self.bucket_size
            buckets[bucket_index].add(sorted_sample[i])
            # When reaching the last node in each bucket store the distances for easy lookup
            if i % self.bucket_size == self.bucket_size - 1:
                self.bucket_distances[pivot].append(self.distance(pivot, sorted_sample[i]))
        # The final bucket may not fill up so make sure the distance is stored
        if len(self.bucket_distances[pivot]) < self.num_buckets:
            self.bucket_distances[pivot].append(self.distance(pivot, sorted_sample[-1]))

    def _place_near_parent(self, index: int, alpha: float) -> None:
        """
        Determine the most similar parent node and place source node
        nearby by comparing distances to other nodes
        """
        source = self.nodes[index]
        parent = self._find_parent(source)
        radius = self.distance(source, parent)
        distance_sum = sum([self.distance(source, pivot) for pivot in self.pivots])

        sample_distances_error = self._create_error_fn(parent, radius, distance_sum)

        lower_angle, upper_angle = self._find_circle_quadrant(sample_distances_error)
        best_angle = self._binary_search_angle(lower_angle, upper_angle, sample_distances_error)
        source.x, source.y = point_on_circle(parent.x, parent.y, best_angle, radius)
        self._force_layout_child(index, alpha)

    def _create_error_fn(self, parent: Node, radius: float,
                         distance_sum: float) -> Callable[[int], float]:
        """
        Create function specific to current node to calculate
        error in distances at an angle on the parent circle
        at the calculated radius
        """

        def sample_distances_error(angle: int):
            point = point_on_circle(parent.x, parent.y, angle, radius)
            return abs(distance_sum - self._pivot_distances_sum(*point))

        return sample_distances_error

    def _find_parent(self, source: Node) -> Node:
        """
        Find the cloest parent node to the source using pivots
        """
        best_fit_buckets = []
        for pivot in self.pivots:
            distance = self.distance(source, pivot)
            bucket_index = self._find_bucket(pivot, distance)
            best_fit_buckets.append(self.pivot_buckets[pivot][bucket_index])

        common = flexible_intersection(*best_fit_buckets)
        parent = min(common, key=lambda p: self.distance(source, p, cache=True))
        return parent

    def _find_bucket(self, pivot: Node, distance: float) -> int:
        for bucket_index in range(self.num_buckets):
            if distance <= self.bucket_distances[pivot][bucket_index]:
                return bucket_index
        return self.num_buckets - 1

    def _force_layout_child(self, index: int, alpha: float) -> None:
        """
        Apply forces from random subsets of the original sample to
        the child node to refine its position
        """
        source = self.nodes[index]
        for i in range(self.remainder_layout_iterations):
            sample_set = random_sample_set(self.random_sample_size, self.sample_set_size, {index})
            for j in sample_set:
                target = self.sample[j]
                x, y = self._current_distance(source, target)
                force = self._force(math.hypot(x, y), self.distance(source, self.sample[j]), alpha)
                source.add_velocity(x * force, y * force)
            source.apply_velocity()

    def _pivot_distances_sum(self, x: float, y: float):
        """
        Sum the total distance from all sample nodes to a point (x, y)
        """
        return sum([math.hypot(target.x - x, target.y - y) for target in self.pivots])
