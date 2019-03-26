from .base_spring_layout import BaseSpringLayout
from .neighbour_sampling import NeighbourSampling
from .node import Node
from ..utils import point_on_circle, random_sample_set, show_progress
from ..distance import euclidean
from typing import Callable, Tuple, List
from enum import Enum
import numpy as np
import math


class HybridStage(Enum):
    LAYOUT_SAMPLE = 0
    LAYOUT_REMAINDER = 1
    REFINE_LAYOUT = 2
    FINISHED = 3


class Hybrid(BaseSpringLayout):
    """
    An implementation of Chalmers, Morrison, and Ross' 2002 hybrid layout.
    It performs the '96 neighbour sampling algorithm, on a sample set, sqrt(n) samples of the data.
    Other data points are then interpolated into the model. Finally, another spring simulation is
    performed to clean up the layout.
    """

    def __init__(self, dataset: np.ndarray = None, nodes: List[Node] = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean,
                 sample_layout_iterations: int = 75, remainder_layout_iterations: int = 5,
                 refine_layout_iterations: int = 5, random_sample_size: int = 15,
                 preset_sample: List[int] = None, target_node_speed: float=0.0,
                 enable_cache: bool = True) -> None:
        iterations = sample_layout_iterations + refine_layout_iterations + 1
        super().__init__(dataset=dataset, nodes=nodes, distance_fn=distance_fn,
                         iterations=iterations, target_node_speed=target_node_speed,
                         enable_cache=enable_cache)
        n = len(self.nodes)
        self.sample_set_size:             int = (len(preset_sample) if preset_sample is not None
                                                 else round(math.sqrt(n)))
        self.random_sample_size:          int = random_sample_size
        self.sample_layout_iterations:    int = sample_layout_iterations
        self.remainder_layout_iterations: int = remainder_layout_iterations
        self.refine_layout_iterations:    int = refine_layout_iterations
        self.sample_indexes:        List[int] = (preset_sample if preset_sample is not None else
                                                 random_sample_set(self.sample_set_size,
                                                                   len(self.nodes)))
        self.remainder_indexes:     List[int] = list(set(range(n)) - set(self.sample_indexes))
        self.sample:               List[Node] = [self.nodes[i] for i in self.sample_indexes]
        self.remainder:            List[Node] = [self.nodes[i] for i in self.remainder_indexes]
        self.data_size_factor:          float = 1 / self.random_sample_size
        self.stage:               HybridStage = HybridStage.LAYOUT_SAMPLE
        self.stages = {
            HybridStage.LAYOUT_SAMPLE: self._sample_layout_stage,
            HybridStage.LAYOUT_REMAINDER: self._remainder_layout_stage,
            HybridStage.REFINE_LAYOUT: self._refine_layout_stage,
        }
        remainder_iters = self.sample_layout_iterations + 1
        refine_iters = remainder_iters + self.refine_layout_iterations
        self.cutoffs = {
            HybridStage.LAYOUT_SAMPLE: self.sample_layout_iterations,
            HybridStage.LAYOUT_REMAINDER: remainder_iters,
            HybridStage.REFINE_LAYOUT: refine_iters,
        }
        self.sample_layout = NeighbourSampling(nodes=self.sample,
                                               distance_fn=self.distance_fn,
                                               iterations=self.sample_layout_iterations,
                                               target_node_speed=target_node_speed,
                                               enable_cache=enable_cache)
        self.refine_layout = NeighbourSampling(nodes=self.nodes,
                                               distance_fn=self.distance_fn,
                                               iterations=self.refine_layout_iterations,
                                               target_node_speed=target_node_speed,
                                               enable_cache=False)

    def average_speed(self) -> float:
        if self.stage == HybridStage.LAYOUT_SAMPLE or self.stage == HybridStage.LAYOUT_REMAINDER:
            return np.inf  # not all nodes are in the layout yet so cannot be finished
        return self.refine_layout.average_speed()

    def _next_stage(self) -> HybridStage:
        return HybridStage(self.stage.value + 1)

    def _completed_stage(self) -> bool:
        return self.stage is not HybridStage.FINISHED and self._i >= self.cutoffs[self.stage]

    def _spring_layout(self, alpha: float = 1) -> None:
        """
        Perform one iteration of the spring layout
        """
        while self._completed_stage():
            self.stage = self._next_stage()
        if self.stage in self.stages:
            self.stages[self.stage](alpha)

    def _sample_layout_stage(self, alpha: float) -> None:
        """
        Layout subset of size sqrt(n) of nodes using '96 neighbour
        sampling algorithm
        """
        show_progress(self._i, self.sample_layout_iterations, stage='Sample')
        self.sample_layout.spring_layout(alpha=alpha, return_after=1)

    def _remainder_layout_stage(self, alpha: float) -> None:
        """
        Layout the remainder nodes close to similar nodes from the
        sample set
        """
        number_remainders = len(self.remainder_indexes)
        for i in range(number_remainders):
            show_progress(i, number_remainders, stage='Remainder')
            self._place_near_parent(self.remainder_indexes[i], alpha)

    def _refine_layout_stage(self, alpha: float) -> None:
        """
        Perform neighbour sampling algorithm on whole layout to refine it
        """
        show_progress(self._i - self.sample_layout_iterations - 1, self.refine_layout_iterations,
                      stage='Refine')
        self.refine_layout.spring_layout(alpha=alpha, return_after=1)

    def _place_near_parent(self, index: int, alpha: float) -> None:
        """
        Determine the most similar parent node and place source node
        nearby by comparing distances to other nodes
        """
        source = self.nodes[index]
        parent_index, distances = self._find_parent(source)
        radius = distances[parent_index]
        parent = self.sample[parent_index]

        sample_distances_error = self._create_error_fn(parent_index, distances)

        lower_angle, upper_angle = self._find_circle_quadrant(sample_distances_error)
        best_angle = self._binary_search_angle(lower_angle, upper_angle, sample_distances_error)
        source.x, source.y = point_on_circle(parent.x, parent.y, best_angle, radius)
        self._force_layout_child(index, distances, alpha)

    def _create_error_fn(self, parent_index: int,
                         distances: List[float]) -> Callable[[int], float]:
        """
        Create function specific to current node to calculate
        error in distances at an angle on the parent circle
        at the calculated radius
        """
        radius = distances[parent_index]
        parent = self.sample[parent_index]
        distance_sum = sum(distances)

        def sample_distances_error(angle: int) -> float:
            point = point_on_circle(parent.x, parent.y, angle, radius)
            return abs(distance_sum - self._sample_distances_sum(*point))

        return sample_distances_error

    def _find_parent(self, source: Node) -> Tuple[int, List[float]]:
        """
        Find the cloest parent node to the source, also return the
        list of calculated distances to speed up later calculation
        """
        distances: List[float] = [self.distance(source, target) for target in self.sample]
        parent_index: int = np.argmin(distances)
        return parent_index, distances

    def _find_circle_quadrant(self, error_fn: Callable[[int], float]) -> Tuple[int, int]:
        """
        Find the quadrant with the minimum error function at the edges
        """
        angles = [0, 90, 180, 270]
        distance_errors = [error_fn(angle) for angle in angles]
        # determine angle with lowest error and choose neighbour quadrant angle with lowest error
        best_angle_i = np.argmin(distance_errors)
        neighbour_angle_indexes = (best_angle_i - 1) % 4, (best_angle_i + 1) % 4
        closest_neighbour = np.argmin([distance_errors[i] for i in neighbour_angle_indexes])
        lower_bound_angle = (
            angles[best_angle_i] if closest_neighbour == 1 else angles[(best_angle_i - 1) % 4]
        )
        return lower_bound_angle, lower_bound_angle + 90

    def _binary_search_angle(self, lower_angle: int, upper_angle: int,
                             error_fn: Callable[[int], float]) -> int:
        """
        Recursively binary search to find the best angle between upper and lower
        bound that minimises the error function
        """
        if upper_angle - lower_angle <= 4:
            return lower_angle  # found a good angle
        angle_range = upper_angle - lower_angle
        lower_error = error_fn(lower_angle + angle_range // 4)
        upper_error = error_fn(upper_angle - angle_range // 4)
        if lower_error < upper_error:
            return self._binary_search_angle(lower_angle, upper_angle - angle_range // 2, error_fn)
        elif upper_error < lower_error:
            return self._binary_search_angle(lower_angle + angle_range // 2, upper_angle, error_fn)
        # both sides are equal so search in the middle
        return self._binary_search_angle(
            lower_angle + angle_range // 4, upper_angle - angle_range // 4, error_fn
        )

    def _force_layout_child(self, index: int, distances: List[float], alpha: float) -> None:
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
                force = self._force(math.hypot(x, y), distances[j], alpha)
                source.add_velocity(x * force, y * force)
            source.apply_velocity()

    def _sample_distances_sum(self, x: float, y: float):
        """
        Sum the total distance from all sample nodes to a point (x, y)
        """
        return sum([math.hypot(target.x - x, target.y - y) for target in self.sample])
