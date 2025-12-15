"""
Copyright (c) 2020-2021 Alexander Lambert

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.autograd import Function


class BaseKernel(ABC):

    @abstractmethod
    def eval(self, X, Y, **kwargs):
        """
        Evaluate kernel function and corresponding gradient terms for batch of inputs.

        Parameters
        ----------
        X : Tensor
            Data, of shape [batch_X, dim]
        Y : Tensor
            Data, of shape [batch_Y, dim]
        kwargs : dict
            Kernel-specific parameters

        Returns
        -------
        K: Tensor
            Kernel Gram matrix, of shape [batch_X, batch_Y].
        d_K_Xi: Tensor
            Kernel gradients wrt. first input X. Shape: [batch_X, batch_Y, dim]
        pw_dists_sq: Tensor (Optional)
            If applicable, returns the squared, inter-particle pairwise distances.
            Shape: [batch_X, batch_Y]
        """
        pass


class RBF(BaseKernel):
    """
        k(x, x') = exp( - || x - x'||**2 / (2 * ell**2))
    """
    # def __init__(
    #     self,
    # ):
    #     super().__init__()

    def eval(
            self,
            X, Y,
            gamma=None,
            **kwargs,
    ):

        assert X.shape[-1] == Y.shape[-1]
        dim = X.shape[-1]
        batch_X = X.shape[0]
        batch_Y = Y.shape[0]

        X = X.reshape(batch_X, 1, dim)
        Y = Y.reshape(1, batch_Y, dim)

        diff_XY = X - Y  # batch_X, batch_Y, dim
        pw_dists_sq = (diff_XY**2).sum(-1)

        K = (- gamma * pw_dists_sq).exp()
        d_K_Xi = K.unsqueeze(2) * diff_XY * 2 * gamma

        return (
            K,
            d_K_Xi,
            pw_dists_sq,
        )

#
# class IMQ(BaseKernel):
#     """
#         IMQ Matrix-valued kernel, with metric M.
#         k(x, x') = M^-1 (alpha + (x - y) M (x - y)^T ) ** beta
#     """
#     def __init__(
#         self,
#         alpha=1,
#         beta=-0.5,
#         hessian_scale=1,
#         analytic_grad=True,
#         median_heuristic=True,
#         **kwargs,
#     ):
#
#         self.alpha = alpha
#         self.beta = beta
#
#         super().__init__(
#             analytic_grad,
#         )
#         self.hessian_scale = hessian_scale
#         self.median_heuristic = median_heuristic
#
#     def eval(
#         self,
#         X, Y,
#         M=None,
#         **kwargs,
#         ):
#
#         assert X.shape == Y.shape
#         b, dim = X.shape
#
#         # Empirical average of Hessian / Fisher matrices
#         M = M.mean(dim=0)
#
#         # PSD stabilization
#         # M = 0.5 * (M + M.T)
#
#         M *= self.hessian_scale
#         X_M_Xt = X @ M @ X.t()
#         X_M_Yt = X @ M @ Y.t()
#         Y_M_Yt = Y @ M @ Y.t()
#
#         pw_dists_sq = -2 * X_M_Yt + X_M_Xt.diag().unsqueeze(1) + Y_M_Yt.diag().unsqueeze(0)
#         if self.median_heuristic:
#             h = torch.median(pw_dists_sq).detach()
#             h = h / np.log(X.shape[0])
#             # h *= 0.5
#         else:
#             h = self.hessian_scale * X.shape[1]
#
#         # Clamp bandwidth
#         tol = 1e-5
#         if isinstance(h, torch.Tensor):
#             h = torch.clamp(h, min=tol)
#         else:
#             h = np.clip(h, a_min=tol, a_max=None)
#
#         K = ( self.alpha + pw_dists_sq) ** self.beta
#         d_K_Xi = self.beta * ((self.alpha + pw_dists_sq) ** (self.beta - 1)).unsqueeze(2) \
#                  * ( -1 * (X.unsqueeze(1) - Y) @ M ) * 2 / h
#                  # * ( (X.unsqueeze(1) - Y) @ M ) * 2 / h
#
#         return (
#             K,
#             d_K_Xi,
#             pw_dists_sq,
#         )
#
#
# class RBF_Anisotropic(RBF):
#     """
#         k(x, x') = exp( - (x - y) M (x - y)^T / (2 * d))
#     """
#     def __init__(
#         self,
#         hessian_scale=1,
#         analytic_grad=True,
#         median_heuristic=False,
#         **kwargs,
#     ):
#         super().__init__(
#             analytic_grad,
#         )
#         self.hessian_scale = hessian_scale
#         self.median_heuristic = median_heuristic
#
#     def eval(
#         self,
#         X, Y,
#         M=None,
#         bw=None,
#         **kwargs,
#     ):
#
#         assert X.shape == Y.shape
#
#         # Empirical average of Hessian / Fisher matrices
#         M = M.mean(dim=0)
#
#         # PSD stabilization
#         # M = 0.5 * (M + M.T)
#
#         M *= self.hessian_scale
#
#         X_M_Xt = X @ M @ X.t()
#         X_M_Yt = X @ M @ Y.t()
#         Y_M_Yt = Y @ M @ Y.t()
#
#         if self.analytic_grad:
#             if self.median_heuristic:
#                 bandwidth, pw_dists_sq = self.compute_bandwidth(X, Y)
#             else:
#                 # bandwidth = self.hessian_scale * X.shape[1]
#                 bandwidth = self.hessian_scale
#                 pw_dists_sq = -2 * X_M_Yt + X_M_Xt.diag().unsqueeze(1) + Y_M_Yt.diag().unsqueeze(0)
#
#             if bw is not None:
#                 bandwidth = bw
#
#             K = (- pw_dists_sq / bandwidth).exp()
#             d_K_Xi = K.unsqueeze(2) * ( (X.unsqueeze(1) - Y) @ M ) * 2 / bandwidth
#         else:
#             raise NotImplementedError
#
#         return (
#             K,
#             d_K_Xi,
#             pw_dists_sq,
#         )
#
#
# class Linear(BaseKernel):
#     """
#         k(x, x') = x^T x' + 1
#     """
#     def __init__(
#         self,
#         analytic_grad=True,
#         subtract_mean=True,
#         with_scaling=False,
#         **kwargs,
#     ):
#         super().__init__(
#             analytic_grad,
#         )
#         self.analytic_grad = analytic_grad
#         self.subtract_mean = subtract_mean
#         self.with_scaling = with_scaling
#
#     def eval(
#             self,
#             X, Y,
#             M=None,
#             **kwargs,
#     ):
#
#         assert X.shape == Y.shape
#         batch, dim = X.shape
#
#         if self.subtract_mean:
#             mean = X.mean(0)
#             X = X - mean
#             Y = Y - mean
#
#         if self.analytic_grad:
#             K = X @ Y.t() + 1
#             d_K_Xi = Y.repeat(batch, 1, 1)
#         else:
#             raise NotImplementedError
#
#         if self.with_scaling:
#             K = K / (dim + 1)
#             d_K_Xi = d_K_Xi / (dim + 1)
#
#         return (
#             K,
#             d_K_Xi,
#             None,
#         )
