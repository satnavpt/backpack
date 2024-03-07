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


class GGNMPBatchNormNd(GGNMPBase):
    def __init__(self):
        super().__init__(
            derivatives=BatchNormNdDerivatives(),
            params=["weight", "bias"],
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        """Compute the GGNMP quantity for the 'weight' parameter."""
        h_out_prod = backproped

        def weight_ggnmp(vector):
            """Matrix-vector product with the GGN matrix w.r.t. the 'weight' parameter."""
            result = self.derivatives._weight_jac_mat_prod(module, g_inp, g_out, vector)
            result = h_out_prod(result)
            result = self.derivatives._weight_jac_t_mat_prod(
                module, g_inp, g_out, result
            )
            return result

        return weight_ggnmp

    def bias(self, ext, module, g_inp, g_out, backproped):
        """Compute the GGNMP quantity for the 'bias' parameter."""
        h_out_prod = backproped

        def bias_ggnmp(vector):
            """Matrix-vector product with the GGN matrix w.r.t. the 'bias' parameter."""
            result = self.derivatives._bias_jac_mat_prod(module, g_inp, g_out, vector)
            result = h_out_prod(result)
            result = self.derivatives._bias_jac_t_mat_prod(module, g_inp, g_out, result)
            return result

        return bias_ggnmp
