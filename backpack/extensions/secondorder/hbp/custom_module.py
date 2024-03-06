"""Module extensions for custom properties of HBPBaseModule."""

from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.backprop_extension import BackpropExtension

from backpack.utils.errors import batch_norm_raise_error_if_train


class HBPScaleModule(HBPBaseModule):
    """HBP extension for ScaleModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())


class HBPSumModule(HBPBaseModule):
    """HBP extension for SumModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=SumModuleDerivatives())


class HBPBatchNormNd(HBPBaseModule):
    """HBP extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["weight", "bias"]
        )

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: BatchNorm1d | BatchNorm2d | BatchNorm3d,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
    ) -> None:
        batch_norm_raise_error_if_train(module)
