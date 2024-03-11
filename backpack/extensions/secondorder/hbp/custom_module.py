"""Module extensions for custom properties of HBPBaseModule."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp.hbp_options import (
    BackpropStrategy,
    ExpectationApproximation,
)
from backpack.utils.errors import batch_norm_raise_error_if_train

from typing import List, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backpack.extensions.secondorder.hbp import HBP

from torch import Tensor, einsum


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
        self._conv_dim = 2
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["weight", "bias"]
        )

    def weight(
        self,
        ext: HBP,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> List[Tensor]:
        """Compute the Kronecker factors for the weight Hessian approximation."""
        # self._maybe_raise_groups_not_implemented_error(ext, module)

        kron_factors: List[Tensor] = []
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):  # KFRA
            kron_factors.append(self._factor_from_batch_average(module, backproped))

        elif BackpropStrategy.is_sqrt(bp_strategy):  # KFLR, KFAC
            kron_factors.append(self._factor_from_sqrt(module, backproped))

        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(
        self, ext: HBP, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d]
    ) -> List[Tensor]:
        """Compute the un-centered covariance of the unfolded input."""
        ea_strategy = ext.get_ea_strategy()
        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError("Undefined")

        # X = convUtils.unfold_input(module, module.input0)
        X = module.input0.flatten(-2)

        return [einsum("bik,bjk->ij", X, X) / X.shape[0]]

    def _factor_from_sqrt(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d], backproped: Tensor
    ) -> Tensor:
        """Compute the Kronecker factor from the backpropagated GGN matrix square root."""
        sqrt_ggn = backproped.flatten(start_dim=-self._conv_dim)
        sqrt_ggn = einsum("cbij->cbi", sqrt_ggn)
        return einsum("cbi,cbl->il", sqrt_ggn, sqrt_ggn)

    def bias(
        self,
        ext: HBP,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> List[Tensor]:
        """Compute the Kronecker factors for the bias Hessian approximation."""
        kron_factors: List[Tensor] = []
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):  # KFRA
            kron_factors.append(self._factor_from_batch_average(module, backproped))

        elif BackpropStrategy.is_sqrt(bp_strategy):  # KFAC/KFLR
            kron_factors.append(self._factor_from_sqrt(module, backproped))

        return kron_factors

    def _factor_from_batch_average(
        self, module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d], backproped: Tensor
    ) -> Tensor:
        """Compute the Kronecker factor from the backpropagated output Hessian proxy."""
        spatial_dim = module.output.shape[-self._conv_dim :].numel()
        out_channels = module.output.shape[-self._conv_dim - 1]

        # sum over spatial coordinates
        return (
            backproped.view(out_channels, spatial_dim, out_channels, spatial_dim)
            .sum([1, 3])
            .contiguous()
        )

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: BatchNorm1d | BatchNorm2d | BatchNorm3d,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
    ) -> None:
        batch_norm_raise_error_if_train(module)
