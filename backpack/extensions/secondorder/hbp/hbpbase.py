from backpack.extensions.curvature import Curvature
from backpack.extensions.module_extension import ModuleExtension
from backpack.extensions.secondorder.hbp.hbp_options import BackpropStrategy
from collections.abc import Callable
from torch.nn import Module
from torch import Tensor


class HBPBaseModule(ModuleExtension):
    def __init__(self, derivatives, params=None, sum_batch=False):
        self.derivatives = derivatives
        if params is not None:
            for param in params:
                if not hasattr(self, param):
                    setattr(self, param, self._make_param_method(param, sum_batch))
        super().__init__(params=params)

    def _make_param_method(
        self, param_str: str, sum_batch: bool
    ) -> Callable[
        [ModuleExtension, Module, tuple[Tensor], tuple[Tensor], Tensor], Tensor
    ]:
        def _param(
            ext: ModuleExtension,
            module: Module,
            grad_inp: tuple[Tensor],
            grad_out: tuple[Tensor],
            backproped: Tensor,
        ) -> Tensor:
            """Returns diagonal of GGN.

            Args:
                ext: extension
                module: module through which to backpropagate
                grad_inp: input gradients
                grad_out: output gradients
                backproped: backpropagated information

            Returns:
                    diagonal
            """
            axis: tuple[int] = (0, 1) if sum_batch else (0,)
            return (
                self.derivatives.param_mjp(
                    param_str, module, grad_inp, grad_out, backproped, sum_batch=False
                )
                ** 2
            ).sum(axis=axis)

        return _param

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self.backpropagate_batch_average(
                ext, module, g_inp, g_out, backproped
            )

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self.backpropagate_sqrt(ext, module, g_inp, g_out, backproped)

    def backpropagate_sqrt(self, ext, module, g_inp, g_out, H):
        return self.derivatives.jac_t_mat_prod(module, g_inp, g_out, H)

    def backpropagate_batch_average(self, ext, module, g_inp, g_out, H):
        ggn = self.derivatives.ea_jac_t_mat_jac_prod(module, g_inp, g_out, H)

        residual = self.second_order_module_effects(module, g_inp, g_out)
        residual_mod = Curvature.modify_residual(residual, ext.get_curv_type())

        if residual_mod is not None:
            ggn = self.add_diag_to_mat(residual_mod, ggn)

        return ggn

    def second_order_module_effects(self, module, g_inp, g_out):
        if self.derivatives.hessian_is_zero(module):
            return None

        elif not self.derivatives.hessian_is_diagonal(module):
            raise NotImplementedError(
                "Residual terms are only supported for elementwise functions"
            )

        else:
            return self.derivatives.hessian_diagonal(module, g_inp, g_out).sum(0)

    @staticmethod
    def add_diag_to_mat(diag, mat):
        assert len(diag.shape) == 1
        assert len(mat.shape) == 2
        assert diag.shape[0] == mat.shape[0] == mat.shape[1]

        dim = diag.shape[0]
        idx = list(range(dim))

        mat[idx, idx] = mat[idx, idx] + diag
        return mat
