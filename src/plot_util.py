import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def shift_plot(
    ys: list[ArrayLike],
    shift: float,
    names: list[str],
    x: ArrayLike | None = None,
    ax=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    for i, (name, y) in enumerate(zip(names, ys)):
        if type(y) is not np.ndarray:
            y = np.asarray(y)
        if x is None:
            ax.plot(y + i * -shift, label=name, **kwargs)
        else:
            ax.plot(x, y + i * -shift, label=name, **kwargs)

    num_ticks = len(names)
    y_ticks = np.linspace(0, -shift * num_ticks, num_ticks, endpoint=False)
    ax.set_yticks(ticks=y_ticks, labels=names)

    return ax
