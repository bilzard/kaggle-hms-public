import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def format_time(x, pos):
    sgn = "-" if x < 0 else ""
    total_seconds = int(abs(x))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{sgn}{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_min(x, pos):
    sgn = "-" if x < 0 else ""
    total_seconds = int(abs(x))
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{sgn}{minutes:02d}:{seconds:02d}"


def shift_plot(
    ys: list[ArrayLike],
    shift: float,
    names: list[str],
    x: ArrayLike | None = None,
    ax=None,
    area=False,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    for i, (name, y) in enumerate(zip(names, ys)):
        if type(y) is not np.ndarray:
            y = np.asarray(y)

        offset = i * -shift
        if area:
            if x is None:
                x = np.arange(len(y))
            ax.fill_between(x, y1=y + offset, y2=offset, label=name, **kwargs)
        else:
            args = [y + offset]
            if x is not None:
                args = [x, *args]
            ax.plot(*args, label=name, **kwargs)

    num_ticks = len(names)
    y_ticks = np.linspace(0, -shift * num_ticks, num_ticks, endpoint=False)
    ax.set_yticks(ticks=y_ticks, labels=names)

    return ax
