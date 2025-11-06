"""Custom abstract base classes for the `Bijection` and `Bijection` types to 
extend flowjax.bijections."""
from collections.abc import Callable

from jax import nn, vmap, numpy as jnp

import equinox as eqx
from paramax import AbstractUnwrappable, NonTrainable
from flowjax.bijections import AbstractBijection

from ecg.utils import kron_block

from typing import ClassVar
from jaxtyping import Array, PRNGKeyArray


class StochasticLinearBijection(AbstractBijection):
    """Stochastic linear bijection with softmax constraint on rows.
    The linear transformation matrix :math:`A` is parameterized as a
    softmax over unconstrained parameters :math:`A'`:
  
    .. math::
        A_{ij} = \frac{\exp(A'_{ij})}{\sum_{k} \exp(A'_{ik})}.

    This ensures that each row of :math:`A` sums to one.

    Args:
        init: Initial unconstrained parameters for the linear transformation.
    """

    params: Array
    shape: tuple[int, ...]
    cond_shape: None = None

    def __init__(self,
                 init: Array):
        assert init.ndim==2 and init.shape[0]==init.shape[1], \
                f'Init must be square matrix, got {init.shape}'
        self.params = init
        self.shape = (init.shape[1],)
        self.cond_shape = None

    @property
    def matrix(self):
        """Compute the stochastic matrix A with softmax row normalization."""
        return nn.softmax(self.params, axis=1)

    @property
    def inv_matrix(self):
        return jnp.linalg.inv(self.matrix)

    def transform_and_log_det(self,
                              x: Array,
                              condition=None
                              ) -> tuple[Array, Array]:
        y = self.matrix @ x
        log_det = self._log_abs_det()
        return y, log_det

    def inverse_and_log_det(self,
                            y: Array,
                            condition=None
                            ) -> tuple[Array, Array]:
        x = jnp.linalg.solve(self.matrix, y)
        log_det = -self._log_abs_det()
        return x, log_det

    def _log_abs_det(self) -> Array:
        logdet = jnp.linalg.slogdet(self.matrix)[1]
        return logdet


class AffineMLP(AbstractBijection):
    """Conditional affine transformation ``y = a*x + b``, 
    where loc ``b`` and scale ``a`` are predicted by an MLP.

    Args:
        key: Jax random key.
        dim: Dimension of the output distribution.
        cond_dim: Dimension of the conditioning input.
        nn_width: Width of the neural networks.
        nn_depth: Depth of the neural networks.
        nn_activation: Activation function for the neural networks.
                       Defaults to `nn.relu`.

    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    dim: int
    conditioner: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        cond_dim: int,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = nn.relu,
    ):
        self.dim = dim
        self.shape = (dim,)
        self.cond_shape = (cond_dim,)

        self.conditioner = eqx.nn.MLP(
                in_size=(
                    cond_dim
                ),
                out_size=dim * 2,
                width_size=nn_width,
                depth=nn_depth,
                activation=nn_activation,
                key=key,
                )

    def transform_and_log_det(self, x, condition):
        loc, log_scale = self.loc_and_log_scale(condition)
        return x * jnp.exp(log_scale) + loc, log_scale.sum()

    def inverse_and_log_det(self, y, condition):
        loc, log_scale = self.loc_and_log_scale(condition)
        return (y - loc) / jnp.exp(log_scale), - log_scale.sum()   

    def loc_and_log_scale(self, condition):
        affine_params = self.conditioner(condition)
        loc, log_scale = affine_params[: self.dim], affine_params[self.dim :]
        return loc, log_scale


class StochasticLinearBijectionPerAtom(AbstractBijection):
    """
    Stochastic linear bijection with softmax constraint on rows, applied per 
    atom.
    """

    params: Array
    shape: tuple[int, ...]
    identity: Array | AbstractUnwrappable[Array]
    cond_shape: None = None
    remove_rows_idxs: Array | None = None

    #TODO: future: add a way to directly import and use matrix from fix_coords
    #TODO: combine with other StochasticLinearBijection, add docstring
    #TODO: check shape init, remove idxs
    def __init__(self,
                 init: Array,
                 remove_rows_idxs: Array | None = None
                 ):
        assert init.ndim==2

        self.params = init
        self.identity = NonTrainable(jnp.eye(3))
        self.shape = (init.shape[1]*3,)
        self.remove_rows_idxs = remove_rows_idxs
        self.cond_shape = None

    @property
    def matrix(self):
        coeffs = nn.softmax(self.params, axis=1)
        coeffs = jnp.stack(coeffs)
        blocks = vmap(kron_block, (0,None))(coeffs, self.identity)
        blocks = blocks.reshape(-1, self.shape[0])
        if self.remove_rows_idxs is not None:
            blocks = jnp.delete(blocks,
                                self.remove_rows_idxs,
                                axis=0,
                                assume_unique_indices=True)
        return blocks

    def transform_and_log_det(self,
                              x: Array,
                              condition=None
                              ) -> tuple[Array, Array]:
        y = self.matrix @ x
        log_det = self._log_abs_det()
        return y, log_det

    def inverse_and_log_det(self,
                            y: Array,
                            condition=None
                            ) -> tuple[Array, Array]:
        x = jnp.linalg.solve(self.matrix, y)
        log_det = -self._log_abs_det()
        return x, log_det

    def _log_abs_det(self) -> Array:
        logdet = jnp.linalg.slogdet(self.matrix)[1]
        return logdet


class SignFlip(AbstractBijection):
    """Elementwise sign-flip bijection: y = -x.
    
    Args:
        shape: Shape of the bijection. Defaults to ().
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        return -x, jnp.zeros(())

    def inverse_and_log_det(self, y, condition=None):
        return -y, jnp.zeros(())
