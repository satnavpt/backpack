"""Module extensions for custom properties of GGNMPBaseModule."""

from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPScaleModule(GGNMPBase):
    """GGNMP extension for ScaleModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())


class GGNMPSumModule(GGNMPBase):
    """GGNMP extension for SumModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=SumModuleDerivatives())
