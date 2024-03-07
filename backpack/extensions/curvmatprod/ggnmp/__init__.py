"""
Matrix-free multiplication with the block-diagonal generalized Gauss-Newton/Fisher.
"""

from collections.abc import Callable

from torch.nn import (
    AvgPool2d,
    BatchNorm1d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
)

from backpack.extensions.secondorder.base import SecondOrderBackpropExtension
from backpack.custom_module.branching import SumModule

from . import (
    activations,
    batchnorm1d,
    conv2d,
    dropout,
    flatten,
    linear,
    losses,
    padding,
    pooling,
    custom_module,
)


class GGNMP(SecondOrderBackpropExtension):
    """
    Matrix-free Multiplication with the block-diagonal generalized Gauss-Newton/Fisher.

    Stores the multiplication function in :code:`ggnmp`.

    For a parameter of shape ``[...]`` the function receives and returns a tensor of
    shape ``[V, ...]``. Each vector slice across the leading dimension is multiplied
    with the block-diagonal GGN/Fisher.
    """

    def __init__(self, savefield="ggnmp"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.GGNMPMSELoss(),
                CrossEntropyLoss: losses.GGNMPCrossEntropyLoss(),
                Linear: linear.GGNMPLinear(),
                MaxPool2d: pooling.GGNMPMaxpool2d(),
                AvgPool2d: pooling.GGNMPAvgPool2d(),
                ZeroPad2d: padding.GGNMPZeroPad2d(),
                Conv2d: conv2d.GGNMPConv2d(),
                Dropout: dropout.GGNMPDropout(),
                Flatten: flatten.GGNMPFlatten(),
                ReLU: activations.GGNMPReLU(),
                Sigmoid: activations.GGNMPSigmoid(),
                Tanh: activations.GGNMPTanh(),
                BatchNorm1d: batchnorm1d.GGNMPBatchNorm1d(),
                SumModule: custom_module.GGNMPSumModule(),
                AdaptiveAvgPool1d: pooling.GGNMPAdaptiveAvgPoolNd(1),
                AdaptiveAvgPool2d: pooling.GGNMPAdaptiveAvgPoolNd(2),
                AdaptiveAvgPool3d: pooling.GGNMPAdaptiveAvgPoolNd(3),
                BatchNorm1d: custom_module.GGNMPBatchNormNd(),
                BatchNorm2d: custom_module.GGNMPBatchNormNd(),
                BatchNorm3d: custom_module.GGNMPBatchNormNd(),
            },
        )

    def accumulate_backpropagated_quantities(
        self, existing: Callable, other: Callable
    ) -> Callable:
        return lambda mat: existing(mat) + other(mat)
