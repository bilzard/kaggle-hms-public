import numpy as np
import torch


class BaseTransform:
    """
    datasetに対して確率的な変換を適用するための基底クラス
    """

    def __init__(self, params: dict, p: float = 1.0):
        self.params = params
        self.p = p

    def __call__(self, feature: np.ndarray, mask: np.ndarray):
        if torch.rand((1,)).item() < self.p:
            feature, mask = self.apply(feature, mask)
        return feature, mask

    def apply(
        self, feature: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __repr__(self):
        params = dict(**self.params, prob=self.p)
        return f"{type(self).__name__}({', '.join([f'{k}={v}' for k, v in params.items()])})"


class Compose(BaseTransform):
    """
    複数のBaseTransformのリストを受け取り、それらを順番に適用する
    """

    def __init__(self, transforms: list[BaseTransform], p: float = 1.0):
        super().__init__(params=dict(), p=p)
        self.transforms = transforms

    def apply(
        self, feature: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            feature, mask = transform(feature, mask)
        return feature, mask

    def __repr__(self):
        reprs = ",\n\t".join([str(t) for t in self.transforms])
        return f"Compose([\n\t{reprs},\n]"
