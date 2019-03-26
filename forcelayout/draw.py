from .algorithms import BaseSpringLayout, Node, NeighbourSampling, Hybrid, Pivot
from typing import Callable, Tuple, List, Optional, FrozenSet, Dict
from matplotlib.animation import FuncAnimation as Animation
import matplotlib.pyplot as plt
import numpy as np

ColorMap = plt.cm


class DrawLayout:
    def __init__(self, dataset: np.ndarray, spring_layout: BaseSpringLayout):
        self.dataset: np.ndarray = dataset
        self.spring_layout: BaseSpringLayout = spring_layout

    def draw(self, alpha: float=None, color_by: Callable[[np.ndarray], float]=None,
             color_map: str='viridis', annotate: Callable[[Node, int], str]=None,
             size: float=40, algorithm_highlights: bool=False) -> None:
        """
        Draw the spring layout graph.
        alpha: float in range 0 - 1 for the opacity of points
        color_by: function to represent a node as a single float which will be used to color it
        color_map: string name of matplotlib.pyplot.cm to take colors from when coloring
                   using color_by
        """
        # Get positions of nodes
        pos: np.ndarray = self.spring_layout.get_positions()
        x = pos[:, 0]
        y = pos[:, 1]

        # Color nodes
        colors, cmap = self._get_colors(color_by, color_map)

        # Draw plot
        sc = plt.scatter(x, y, s=size, alpha=alpha, c=colors, cmap=cmap)
        plt.axis('off')

        if annotate is not None:
            self._enable_annotation(annotate=annotate,
                                    scatter=sc,
                                    color_by=color_by)

        if algorithm_highlights:
            self._enable_highlights(scatter=sc)

    def draw_animated(self, alpha: float=None, color_by: Callable[[np.ndarray], float]=None,
                      color_map: str='viridis', interval: int = 10, draw_every: int = 1,
                      annotate: Callable[[Node, int], str]=None, size: float = 40,
                      algorithm_highlights: bool=False) -> Animation:
        """
        Draw an animated spring layout graph.
        alpha: float in range 0 - 1 for the opacity of points
        color_by: function to represent a node as a single float which will be used to color it
        color_map: string name of matplotlib.pyplot.cm to take colors from when coloring
                   using color_by
        interval: minimum milliseconds between updating nodes (usually capped by algorithm
                  performance)
        """
        # Color nodes and remove axis
        colors, cmap = self._get_colors(color_by, color_map)
        plt.axis('off')

        pos = np.zeros((len(self.spring_layout.nodes), 2))

        fig = plt.gcf()
        ax = plt.gca()
        scatter = ax.scatter(pos[:, 0], pos[:, 1], s=size, alpha=alpha, c=colors, cmap=cmap)

        def update_nodes(frame):
            pos = self.spring_layout.spring_layout(return_after=draw_every)
            scatter.set_offsets(pos)
            ax.set_xlim(*self._get_limits(pos[:, 0], delta=5))
            ax.set_ylim(*self._get_limits(pos[:, 1], delta=5))
            return scatter,

        if annotate is not None:
            self._enable_annotation(annotate=annotate,
                                    scatter=scatter,
                                    color_by=color_by)

        if algorithm_highlights:
            self._enable_highlights(scatter=scatter)

        self.ani = Animation(fig, update_nodes, interval=interval,
                             save_count=self.spring_layout.iterations)
        return self.ani

    def save(self, path: str, *args, **kwargs) -> None:
        """ Save the current layout """
        plt.savefig(path, *args, **kwargs)

    def _enable_annotation(self, annotate: Callable[[Node, int], str],
                           color_by, scatter) -> None:
        """
        Enables drawing of annotations when nodes are
        hovered using the annotate callback function
        """
        self._active_annotations: FrozenSet[int] = frozenset()
        fig = plt.gcf()
        ax = plt.gca()
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"),
                            fontsize='16')
        annot.set_visible(False)

        def update_annot(ind):
            indexes: FrozenSet[int] = frozenset(ind["ind"])
            if indexes == self._active_annotations:
                return
            self._active_annotations = indexes
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = ", ".join(
                [annotate(self.spring_layout.nodes[n], n) for n in ind["ind"][:5]]
            )
            if len(ind["ind"]) > 5:
                text += ', ...'
            annot.set_text(text)
            if color_by is not None:
                annot.get_bbox_patch().set_facecolor(scatter._facecolors[ind['ind'][0]])
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                elif vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

    def _enable_highlights(self, scatter) -> None:
        fig = plt.gcf()
        ax = plt.gca()

        self._highlighted: Dict[int, np.ndarray] = dict()

        def reset_colors():
            for index, col in self._highlighted.items():
                scatter._facecolors[index, :] = col
            self._highlighted.clear()

        def click(event):
            reset_colors()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    top_index = ind['ind'][0]
                    highlighted_indexes = self._get_highlighted(top_index)
                    for index in highlighted_indexes:
                        self._highlighted[index] = scatter._facecolors[index, :].copy()
                        scatter._facecolors[index, :] = (1, 0, 0, 1)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', click)

    def _get_highlighted(self, index: int) -> List[int]:
        layout = self.spring_layout
        if type(layout) is NeighbourSampling:
            return layout._get_neighbours(index)
        if type(layout) is Hybrid:
            return layout.sample_indexes
        if type(layout) is Pivot:
            return layout.pivot_indexes
        return []

    def _get_colors(self, color_by: Optional[Callable[[np.ndarray], float]],
                    color_map: str) -> Tuple:
        colors = 'b'
        cmap = None
        if color_by is not None:
            colors = np.apply_along_axis(color_by, axis=1, arr=self.dataset)
            cmap = plt.cm.get_cmap(color_map)
        return (colors, cmap)

    def _get_limits(self, arr: np.ndarray, delta: float = 0.0) -> Tuple[float, float]:
        lower = arr.min()
        upper = arr.max()
        diff = upper - lower
        padding = diff * delta / 100
        return (arr.min() - padding, arr.max() + padding)
