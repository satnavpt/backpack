"""Block generalized Gauss-Newton matrix products"""

from backpack.extensions.module_extension import ModuleExtension
from torch.nn import Module
from torch import Tensor
from typing import Callable


class GGNMPBase(ModuleExtension):
    def __init__(self, derivatives, params=None, sum_batch=False):
        self.derivatives = derivatives
        super().__init__(params=params)

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        """Backpropagate Hessian multiplication routines."""
        h_out_mat_prod = backproped

        def h_in_mat_prod(mat):
            """Multiplication with curvature matrix w.r.t. the module input."""
            # Vectorize the matrix multiplications using torch.matmul or torch.einsum
            result = self.derivatives.jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.jac_t_mat_prod(module, g_inp, g_out, result)
            return result

        return h_in_mat_prod
