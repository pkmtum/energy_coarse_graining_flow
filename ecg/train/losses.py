"""Custom loss functions for training normalizing flows.

For ``fit_to_key_beta_based_loss``, the loss function signature must match
``(params, static, key, beta)``.
"""
from collections.abc import Callable

from jax import vmap

import equinox as eqx
from flowjax.distributions import AbstractDistribution

from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray


class EnergyLoss:
    """KL divergence between joint distributions for KL(q(X,z)||p(X,z)) in 
    https://arxiv.org/pdf/2504.20940, approximated using samples.

    Args:
        num_samples: Number of samples to use in the ELBO approximation.
        energy: The energy, i.e. the potential function / the negative log 
            posterior density up to an additive constant, evaluated for a single 
            point.
        weight_samples: Numer of samples to calculate log weights.
        order_fn: Transformation and log det for coordinate based transform.
    """
    energy: Callable[[ArrayLike], Array]
    num_samples: int
    weight_samples: int
    order_fn: Callable | None

    def __init__(
        self,
        energy: Callable[[ArrayLike], Array],
        num_samples: int,
        *,
        weight_samples: int = 1000,
        order_fn: Callable | None = None,
    ):
        self.energy = energy
        self.order_fn = order_fn
        self.weight_samples = weight_samples
        self.num_samples = num_samples

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        key: PRNGKeyArray,
        beta: Float,
    ) -> Float[Array, ""]:
        """Compute the KL loss."""
        model = eqx.combine(params, static)

        x, joint_log_prob = model.sample_and_log_prob(key,
                                                      (self.num_samples,))

        beta_potential =  beta * vmap(self.energy)(x)

        return (joint_log_prob + beta_potential).mean()

    @eqx.filter_jit
    def compute_log_weights(
        self,
        model: AbstractDistribution,
        key: PRNGKeyArray,
        beta: Float,
        ) -> tuple[Array, Array, Array]:
        """Compute the weighted loss components."""

        x, joint_log_prob = model.sample_and_log_prob(key,
                                                    (self.weight_samples,))

        beta_potential =  beta * vmap(self.energy)(x)

        log_weights = - beta_potential - joint_log_prob
        return beta_potential, joint_log_prob, log_weights
