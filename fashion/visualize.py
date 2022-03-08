import math
from typing import List, Optional

import fastai.vision.all
import matplotlib.pyplot as plt


def show_images(images, titles: Optional[List[str]] = None, title_font_size=10):
    n = len(images)
    plt.rcParams["axes.titlesize"] = title_font_size
    ncols = math.ceil(n ** 0.5)
    nrows = math.ceil(n ** 0.5)
    fastai.vision.all.show_images(
        images, titles=titles, ncols=ncols, nrows=nrows, squeeze=False
    )


def target_to_class(dataset, target) -> str:
    if target < 0 or target > len(dataset.classes) - 1:
        return "Unknown"
    return dataset.classes[target]


def targets_to_classes(dataset, targets) -> List[str]:
    return [target_to_class(dataset, target) for target in targets]
