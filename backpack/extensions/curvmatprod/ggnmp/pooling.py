from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase
from backpack.core.derivatives.adaptive_avg_pool_nd import (
    AdaptiveAvgPoolNDDerivatives,
)


class GGNMPAvgPool2d(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class GGNMPMaxpool2d(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class GGNMPAdaptiveAvgPoolNd(GGNMPBase):
    def __init__(self, N: int):
        super().__init__(derivatives=AdaptiveAvgPoolNDDerivatives(N=N))
