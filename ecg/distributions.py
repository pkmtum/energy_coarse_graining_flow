"""Custom distributions to extend flowjax.distributions."""
from collections.abc import Callable

from jax import random, nn, numpy as jnp
from jax.scipy.stats import truncnorm

import equinox as eqx
import paramax

from flowjax.bijections import AbstractBijection
from flowjax.distributions import AbstractDistribution, AbstractTransformed, \
                                  MultivariateNormal, VmapMixture, \
                                  StandardNormal
from flowjax.utils import arraylike_to_array

from ecg.bijections import AffineMLP

from typing import ClassVar
from jaxtyping import ArrayLike, PRNGKeyArray, Array


class TruncatedNormal(AbstractDistribution):
    """Truncated normal distribution.

    ``minval`` and ``maxval`` should broadcast to the desired 
    distribution shape.

    Args:
        shape: The shape of the distribution.
        minval: Minimum values.
        maxval: Maximum values.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    minval: ArrayLike
    maxval: ArrayLike

    def __init__(self, minval: ArrayLike, maxval: ArrayLike):
        minval, maxval = arraylike_to_array(minval), arraylike_to_array(maxval)
        shape = jnp.broadcast_shapes(minval.shape, maxval.shape)
        minval, maxval = eqx.error_if(
            (minval, maxval), maxval <= minval, \
            f'minval {minval} must be less than the maxval {maxval}.'
        )
        self.shape = shape
        self.minval = minval
        self.maxval = maxval

    def _log_prob(self, x, condition=None):
        return truncnorm.logpdf(x, a=self.minval, b=self.maxval).sum()

    def _sample(self, key, condition=None):
        return random.truncated_normal(key,
                                       lower=self.minval,
                                       upper=self.maxval,
                                       shape=self.shape)


class MLPConditionalNormal(AbstractTransformed):
    """A conditional normal distribution with mean and scale given by
    an MLP.
    
    The condition is passed to the MLP to compute the
    location and scale.

    Args:
         key: Jax random key.
        dim: Dimension of the output distribution.
        cond_dim: Dimension of the conditioning input.
        nn_width: Width of the neural networks.
        nn_depth: Depth of the neural networks.
        nn_activation: Activation function for the neural networks.
                       Defaults to `nn.relu`.
    """

    base_dist: StandardNormal
    bijection: AffineMLP

    def __init__(self,
                 key: PRNGKeyArray,
                 dim: int,
                 cond_dim: int,
                 nn_width: int,
                 nn_depth: int,
                 nn_activation: Callable = nn.relu,
                 ):

        self.base_dist = StandardNormal(
            (dim,),
        )
        self.bijection = AffineMLP(
                        key=key,
                        dim=dim,
                        cond_dim=cond_dim,
                        nn_width=nn_width,
                        nn_depth=nn_depth,
                        nn_activation=nn_activation,
                        )

    def loc(self, condition):
        """Location of the distribution."""
        return self.bijection.loc_and_log_scale(condition)[0]

    def scale(self, condition):
        """Scale of the distribution."""
        return jnp.exp(self.bijection.loc_and_log_scale(condition)[1])


class JointDistribution(AbstractDistribution):
    """Wrapper model that combines marginal and conditional
    distribution into a single interface with .sample and .log_prob."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    marginal_dist: AbstractDistribution
    cond_dist: AbstractDistribution
    z_dim: int

    def __init__(self,
                 marginal_dist: AbstractDistribution,
                 cond_dist: AbstractDistribution,
                 ):
        self.shape = (marginal_dist.shape[0] + cond_dist.shape[0],)
        self.z_dim = marginal_dist.shape[0]
        self.marginal_dist = marginal_dist
        self.cond_dist = cond_dist

    def _sample(self, key: PRNGKeyArray, condition=None) -> Array:
        """Sample from the combined model."""
        marginal_key, cond_key = random.split(key)

        z = self.marginal_dist.sample(marginal_key)
        X = self.cond_dist.sample(cond_key, condition=z)

        zX = self._stack(z, X)
        return zX

    def _log_prob(self, x: Array, condition=None) -> Array:
        """Compute log-probability under the combined model."""
        z, X = self._split(x)

        log_q_prob = self.marginal_dist.log_prob(z) + \
                     self.cond_dist.log_prob(X, condition=z)
        return log_q_prob

    def _split(self, x: Array) -> tuple[Array, Array]:
        """Split input into marginal and conditional parts."""
        z, X = x[..., :self.z_dim], x[..., self.z_dim:]
        return z, X

    def _stack(self, z: Array, X: Array) -> Array:
        """Stack marginal and conditional parts into single array."""
        return jnp.hstack((z, X))

    def sample_and_split(self,
                         key: PRNGKeyArray,
                         sample_shape: tuple[int, ...] = (),
                         condition: ArrayLike | None = None,
                         ) -> tuple[Array, Array]:
        """Sample from the combined model and leave marginal and 
        conditional parts separate."""
        zX = self.sample(key, sample_shape, condition)
        z, X = self._split(zX)
        return z, X


class JointModelTransformed(AbstractTransformed):
    """Flowjax.distribution.Transformed with extra functionality:
        - Option to obtain splitted samples from a transformed joint
        distribution after the bijection.
        - Option to load and save parameters.
        - Accessors for linear transformation matrix A and its inverse A^{-1}.
    """

    base_dist: JointDistribution
    bijection: AbstractBijection

    def sample_and_split(self,
                         key: PRNGKeyArray,
                         sample_shape: tuple[int, ...] = (),
                         condition: ArrayLike | None = None,
                         ) -> tuple[Array, Array]:
        """Sample from the combined model and split transformed marginal
        and conditional parts."""
        x = self.sample(key, sample_shape, condition)
        x_z, x_X = self.base_dist._split(x)
        return x_z, x_X

    def load(self, params_path: str | None):
        """Load parameters from a file."""
        if params_path is not None:
            print('load parameters')
            return eqx.tree_deserialise_leaves(params_path, self)
        else:
            return self

    def save(self, params_path: str | None):
        """Save parameters to a file."""
        if params_path is not None:
            eqx.tree_serialise_leaves(params_path, self)

    @property
    def matrix(self) -> Array:
        """Return the linear transformation matrix A."""
        return paramax.unwrap(self.bijection.bijections[0]).matrix

    @property
    def inv_matrix(self) -> Array:
        """Return the inverse linear transformation matrix A^{-1}."""
        return paramax.unwrap(self.bijection.bijections[0]).inv_matrix


class GaussianMixtureWithParams(VmapMixture):
    """A flowjax Gaussian Mixture Model with convenient accessors."""

    @property
    def means(self):
        """Return component means (n_components, dim)."""
        return self.dist.loc

    @property
    def covariances(self):
        """Return component covariance matrices (n_components, dim, dim)."""
        chol = paramax.unwrap(self.dist.bijection.triangular)
        return chol @ jnp.swapaxes(chol, -1, -2)

    @property
    def weights(self):
        """Return normalized mixture weights (n_components,)."""
        log_w = paramax.unwrap(self.log_normalized_weights)
        return jnp.exp(log_w)


def GaussianMixture(means, covs, weights):
    """
    Create a Gaussian mixture distribution.

    Args:
        means: array of shape (n_components, dim)
        covs: array of shape (n_components, dim, dim) (symmetric, PD)
        weights: array of shape (n_components,), positive weights
    Returns:
        A VmapMixture distribution
    """

    means = arraylike_to_array(means)
    covs = arraylike_to_array(covs)
    weights = arraylike_to_array(weights)

    assert len(means) == len(covs) == len(weights)

    n_components, dim = means.shape

    assert covs.shape == (n_components, dim, dim), \
        f'Expected covs of shape {(n_components, dim, dim)}, got {covs.shape}'

    mvn = eqx.filter_vmap(MultivariateNormal)(means, covs)

    return GaussianMixtureWithParams(dist=mvn, weights=weights)
