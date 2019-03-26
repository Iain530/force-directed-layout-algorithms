from typing import Set, List, Tuple, Callable, Any, DefaultDict
from collections import defaultdict
import random
import math
import sys
import numpy as np


def jiggle() -> float:
    """
    Return a random small non zero number
    """
    small_non_zero = (random.random() - 0.5) * 1e-6
    while small_non_zero == 0:
        small_non_zero = (random.random() - 0.5) * 1e-6
    return small_non_zero


def random_sample_set(size: int, n: int, exclude: Set[int] = None) -> List[int]:
    """
    Randomly sample in the range 0 to n-1 to fill a set of given size.
    """
    sample_set: Set[int] = set()
    exclude = exclude if exclude is not None else set()
    while len(sample_set) < size:
        s = sample(n, exclude)
        sample_set.add(s)
        exclude.add(s)
    return list(sample_set)


def sample(n: int, exclude: Set[int] = None) -> int:
    """
    Return a random number in the the range 0 to n-1
    not in exclude set.
    """
    s = random.randint(0, n - 1)
    if exclude is None or s not in exclude:
        return s
    return sample(n, exclude)


def point_on_circle(x: float, y: float, angle: float, radius: float) -> Tuple[float, float]:
    """
    Find the point on a circle centered at (x, y) with
    given radius and angle
    """
    rad = math.radians(angle)
    return (
        x + radius * math.cos(rad),
        y + radius * math.sin(rad),
    )


def flexible_intersection(*sets: List[Set[Any]]) -> Set:
    """
    Returns the intersection of sets. If the intersection is empty,
    then the most commonly occuring elements are returned
    """
    counts: DefaultDict[Any, int] = defaultdict(lambda: 0)
    largest = 0
    for s in sets:
        for elm in s:
            counts[elm] += 1
            if counts[elm] > largest:
                largest = counts[elm]
    return {k for k in counts if counts[k] == largest}


def get_average_datapoint(datapoints: np.ndarray) -> np.ndarray:
    """
    Find the average datapoint in the dataset
    """
    return np.mean(datapoints, axis=0)


def mean(lst: List[float]) -> float:
    return sum(lst) / len(lst)


def score_n_samples(dataset: np.ndarray, distance_fn: Callable[[np.ndarray, np.ndarray], float],
                    n: int, size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    size = size if size is not None else round(math.sqrt(len(dataset)))
    samples = np.array([random_sample_set(size, len(dataset)) for i in range(n)])
    data_samples = np.apply_along_axis(lambda s: dataset[s], axis=1, arr=samples)
    dataset_average = get_average_datapoint(dataset)
    averages = np.apply_along_axis(get_average_datapoint, axis=1, arr=data_samples)
    distances = np.apply_along_axis(lambda ds: distance_fn(dataset_average, ds),
                                    axis=1, arr=averages)
    order = np.argsort(distances)
    return samples[order], distances[order]


def show_progress(current: int, total: int, stage: str='') -> None:
    if total == 0:
        return
    # Don't show over 100% progress
    percent = min(int(current / total * 100), 100)
    progress = percent // 5
    # if len(stage) < 10:
    #     stage += ' ' * (10 - len(stage))
    print(f'{stage.ljust(10)} [{"=" * progress}{" " * (20 - progress)}] {percent}% ', end='\r')


def get_size(obj: Any, seen: Set[int] = None):
    """
    Recursively finds size of an object in bytes
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen before entering recursion to handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
