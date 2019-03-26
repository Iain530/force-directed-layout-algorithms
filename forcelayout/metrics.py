from .algorithms import BaseSpringLayout, Hybrid
from typing import List, Dict
from time import perf_counter as timer
import numpy as np


class LayoutMetrics:
    def __init__(self, spring_layout: BaseSpringLayout) -> None:
        self.spring_layout: BaseSpringLayout = spring_layout
        self.memory: Dict[int, int] = dict()
        self.n: int = len(self.spring_layout.nodes)
        self.iter_times: List[float] = []
        self.avg_node_speed: List[float] = []
        self.use_checkpoints: bool = issubclass(type(self.spring_layout), Hybrid)
        if self.use_checkpoints:
            self.checkpoints: List[np.ndarray] = []
            self.stage_times: List[List[float]] = [[], [], []]

    def measure(self) -> None:
        current_iter = self.spring_layout.current_iteration()
        for i in range(current_iter, self.spring_layout.iterations):
            start = timer()
            self.spring_layout.spring_layout(return_after=1)
            end = timer()
            self.iter_times.append(end - start)
            self.get_memory(i)
            self.get_avg_speed(i)
            if self.use_checkpoints:
                self.stage_times[self.spring_layout.stage.value].append(end - start)
                if self.spring_layout._completed_stage():
                    self.checkpoints.append(self.spring_layout.get_positions())

    def get_stress(self) -> float:
        return self.spring_layout.get_stress()

    def get_memory(self, i) -> int:
        if i not in self.memory:
            self.memory[i] = self.spring_layout.get_memory()
        return self.memory[i]

    def get_max_memory(self) -> int:
        return max(self.memory.values()) if len(self.memory) > 0 else None

    def get_run_time(self) -> float:
        return sum(self.iter_times)

    def get_avg_speed(self, i) -> float:
        if i not in self.avg_node_speed:
            self.avg_node_speed.append(self.spring_layout.average_speed())
        return self.avg_node_speed[i]

    def get_average_iteration_time(self) -> float:
        if self.spring_layout.current_iteration() <= 0:
            return None
        return sum(self.iter_times) / len(self.iter_times)

    def to_list(self) -> List:
        results = [
            self.n,
            sum(self.iter_times),
            self.avg_node_speed[-1] if len(self.get_avg_speed) > 0 else np.inf,
            self.spring_layout.current_iteration(),
            max(self.memory.values()),
        ]
        if self.use_checkpoints:
            results += [sum(stage_iter_items) for stage_iter_items in self.stage_times]
        return results

    def to_csv(self) -> str:
        return ','.join(map(str, self.to_list()))

    def get_csv_header(self) -> str:
        header = '"N","Total time","Average speed @ end","Iterations","Max Memory Usage"'
        if self.use_checkpoints:
            header += ',"Sample Layout Time","Interpolation Time","Refine Layout Time"'
        return header
