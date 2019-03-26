from .base_spring_layout import BaseSpringLayout
from .node import Node
from ..utils import random_sample_set, show_progress
from ..distance import euclidean
from typing import Callable, List, Dict
import numpy as np


class NeighbourSampling(BaseSpringLayout):
    """
    An implementation of Chalmers' 1996 Neighbour and Sampling algorithm.
    Using random sampling to find the closest neighbours from the data set.
    """
    def __init__(self, dataset: np.ndarray=None, nodes: List[Node]=None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float]=euclidean,
                 iterations: int=50, neighbour_set_size: int=5, sample_set_size: int=10,
                 target_node_speed: float=0.0, enable_cache: bool=True):
        super().__init__(dataset=dataset, nodes=nodes, distance_fn=distance_fn,
                         iterations=iterations, target_node_speed=target_node_speed,
                         enable_cache=enable_cache)
        assert neighbour_set_size > 0, "neighbour_set_size must be > 0"
        assert sample_set_size > 0, "sample_set_size must be > 0"
        self.neighbour_set_size: int = neighbour_set_size
        self.sample_set_size:    int = sample_set_size
        self.neighbours: Dict[int, List[int]] = dict()
        self.data_size_factor: float = 0.5 / (neighbour_set_size + sample_set_size)

    def _spring_layout(self, alpha: float=1) -> None:
        """
        Perform one iteration of the spring layout
        """
        n = len(self.nodes)
        for i in range(n):
            sample_set = self._get_sample_set(i)
            neighbour_set = self._get_neighbours(i)
            for j in sample_set:
                self._set_velocity(self.nodes[i], self.nodes[j], alpha)
            for j in neighbour_set:
                self._set_velocity(self.nodes[i], self.nodes[j], alpha, cache_distance=True)
            self._update_neighbours(i, samples=sample_set)
        self._apply_velocities()
        show_progress(self._i, self.iterations)

    def _get_neighbours(self, index: int) -> List[int]:
        """
        Get the list of neighbour indicies for a given node index sorted by distance.
        If no neighbouts exist yet then they are randomly sampled.
        """
        if index not in self.neighbours:
            random_sample = random_sample_set(self.neighbour_set_size, len(self.nodes), {index})
            random_sample.sort(
                key=lambda j: self.distance(self.nodes[index], self.nodes[j])
            )
            self.neighbours[index] = random_sample
        return self.neighbours[index]

    def _get_sample_set(self, i: int) -> List[int]:
        """
        Get a valid sample set for a node index by randomly sampling, excluding
        current node and neighbours of the node.
        """
        exclude = {i}.union(set(self._get_neighbours(i)))
        return list(random_sample_set(self.sample_set_size, len(self.nodes), exclude))

    def _update_neighbours(self, i: int, samples: List[int]) -> None:
        """
        Update the neighbour set for a given index from a sample set.
        Sample nodes are added to the neighbour set in sorted order if
        they are closer than the furthest current neighbour.
        """
        source = self.nodes[i]
        neighbours = self._get_neighbours(i)
        furthest_neighbour = self.distance(source, self.nodes[neighbours[-1]])
        for s in samples:
            sample_distance = self.distance(source, self.nodes[s])
            if sample_distance < furthest_neighbour:
                n = self.neighbour_set_size - 2
                neighbour_distance = self.distance(source, self.nodes[neighbours[n]])
                while sample_distance < neighbour_distance:
                    n -= 1
                    if n < 0:
                        break
                    neighbour_distance = self.distance(source, self.nodes[neighbours[n]])
                neighbours.insert(n + 1, s)
                distance_key = frozenset({source, self.nodes[neighbours[-1]]})
                if self.enable_cache and distance_key in self.distances:
                    del self.distances[distance_key]  # Remove distance from cache to save memory
                del neighbours[-1]
