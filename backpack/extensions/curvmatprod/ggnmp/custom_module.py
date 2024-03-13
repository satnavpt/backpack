"""Module extensions for custom properties of GGNMPBaseModule."""

from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from backpack.utils.errors import batch_norm_raise_error_if_train


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
