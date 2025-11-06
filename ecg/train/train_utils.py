"""Utility functions for training."""

from collections.abc import Callable
from typing import Any, Optional, Tuple, NamedTuple

from jax import jit, lax, nn, numpy as jnp

import equinox as eqx
import paramax
import optax

from jaxtyping import Array, ArrayLike, Float, PyTree, PRNGKeyArray, Scalar


@jit
def effective_sample_size(log_weights: Array) -> Scalar:
    """Calculates the approximation to the effective sample size given
    the unnormalized log weights.

    Args:
        log_weights: Array of unnormalized log weights.

    Returns:
        Approximate effective sample size as a scalar float array.
    """
    normalized_log_weights = log_weights - nn.logsumexp(log_weights)
    normalized_weights = jnp.exp(normalized_log_weights)
    return 1.0 / jnp.sum(normalized_weights**2)


class TrainState(NamedTuple):
    opt_state: optax.OptState
    losses: list[ArrayLike]
    grad_norm: list[PyTree]
    bij_params: list[PyTree] = []


@eqx.filter_jit
def step_aux(
    params: PyTree,
    *args: Any,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable[[PyTree, Any], Scalar],
    **kwargs: Any,
) -> Tuple[PyTree, PyTree, Scalar, Any]:
    """Flowjax step function with gradient norm info.
    Carry out a training step.

    Args:
        params: Parameters for the model.
        *args: Arguments passed to the loss function (often the static 
            components of the model).
        optimizer: Optax optimizer.
        opt_state: Optimizer state.
        loss_fn: The loss function. This should take params and static as the 
            first two arguments.
        **kwargs: Key word arguments passed to the loss function.

    Returns:
        tuple: (params, opt_state, loss_val, grad_norm)
    """
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, *args, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)

    return params, opt_state, loss_val, optax.global_norm(grads)


def init_array_tempering(array: Array,
                         **unused_kwargs,
                         ) -> Tuple[
                             Callable[[PyTree, PRNGKeyArray, Scalar, int], \
                                        Tuple[Scalar, int]], int
                        ]:
    """Initialize inverse temperature function from an array of betas.

    Returns a tempering function and the initial index (0).
    """
    def _tempering_fn(model: PyTree,
                      key: PRNGKeyArray,
                      beta: Float,
                      idx: int) -> Tuple[Scalar, int]:
        """Allows for model, key, beta to be passed for uniform design
        of tempering functions."""
        _ = (model, key, beta)
        return jnp.round(array[idx], 6), idx

    return _tempering_fn, 0


def init_linear_tempering(delta_beta: Float,
                          **unused_kwargs
                          ) -> Callable[[PyTree, PRNGKeyArray, Scalar, Any], \
                                        Tuple[Scalar, Any]]:
    """init linear stepwise temperature increase."""

    def _tempering_fn(model: PyTree,
                      key: PRNGKeyArray,
                      beta: Float,
                      states: None) -> Tuple[Scalar, None]:
        """Allows for model, key, states to be passed for uniform design
        of tempering functions."""
        _ = (model, key, states)
        next_beta = beta + delta_beta
        return next_beta, None

    return _tempering_fn


def init_tempering(model: PyTree,
                   key: PRNGKeyArray,
                   beta: Float,
                   **kwargs
                   ) -> Tuple[Callable[[PyTree, PRNGKeyArray, Scalar, Any], \
                                       Tuple[Scalar, Any]], Any]:

    method = kwargs.get('method', None)

    if method == 'linear':
        return init_linear_tempering(**kwargs), None

    elif method == 'adaptive':
        adaptive_tempering_fn, log_Z = init_adaptive_tempering(
            model,
            key,
            init_beta=beta,
            **kwargs,
        )

        return adaptive_tempering_fn, log_Z

    elif method == 'array':
        array_tempering_fn, idx = init_array_tempering(**kwargs)

        return array_tempering_fn, idx

    elif method is None:

        def no_tempering_fn(model: PyTree,
                            key: PRNGKeyArray,
                            beta: Float,
                            states: None) -> Tuple[Scalar, None]:
            """Allows for model, key, states to be passed for uniform design
            of tempering functions."""
            _ = (model, key, states)
            return beta, None

        return no_tempering_fn, None

    else:
        raise ValueError(
            f'Method {method} not implemented. '
            'Select "linear", "adaptive", "array" or None.'
        )


def init_adaptive_tempering(
    model: PyTree,
    key: PRNGKeyArray,
    init_beta: Float,
    get_log_weights: Optional[Callable[[PyTree, PRNGKeyArray, Float], \
                            Tuple[Array, Array, Array]]] | None = None,
    detla_kl: float = 1.0,
    delta_beta: Float = 1e-3,
    **unused_kwargs,
    ) -> Tuple[Callable[[PyTree, PRNGKeyArray, Float, Float], \
                        Tuple[Float, Float]], Float]:

    """Adaptive tempering scheme from https://arxiv.org/abs/2002.10148."""

    assert get_log_weights is not None, \
        'get_log_weights_fn must be provided for adaptive tempering.'

    _, _, init_log_weights = get_log_weights(model,
                                             key,
                                             init_beta)

    num_samples = init_log_weights.shape[0]
    init_log_Z = -jnp.log(num_samples) + nn.logsumexp(init_log_weights)

    @eqx.filter_jit
    def adaptive_tempering_fn(model: PyTree,
                              key: PRNGKeyArray,
                              current_beta: Float,
                              current_log_Z: Float) -> Tuple[Float, Float]:
        params, static = eqx.partition(
            model,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        # Calculate loss parts
        beta_potential, model_loss, log_weights = get_log_weights(model,
                                                                  key,
                                                                  current_beta)

        # Calclate beta independent weights
        log_norm_weights = log_weights - nn.logsumexp(log_weights)

        # MC estimate of model params
        beta_potential_mean = jnp.mean(beta_potential)
        model_loss_mean = jnp.mean(model_loss)

        delta_beta_by_beta = delta_beta / current_beta

        init_delta_log_Z = nn.logsumexp(log_norm_weights - delta_beta_by_beta \
                                        * beta_potential)

        # Calculate initial kl increase
        kl_q_p_current_beta = \
            current_log_Z + beta_potential_mean + model_loss_mean
        init_kl = (init_delta_log_Z + delta_beta_by_beta * \
                    beta_potential_mean) / kl_q_p_current_beta

        init_carry = (delta_beta, init_delta_log_Z, init_kl)

        def _body_fn(carry):
            delta_beta, _, _ = carry

            next_delta_beta = delta_beta * 0.6
            next_delta_beta_by_beta = next_delta_beta / current_beta

            # Calculate next delta log Z
            delta_log_Z = nn.logsumexp(log_norm_weights \
                                       - next_delta_beta_by_beta \
                                        * beta_potential)

            # Calculate next kl increase
            next_kl = (delta_log_Z + next_delta_beta_by_beta * \
                       beta_potential_mean) / kl_q_p_current_beta

            return (next_delta_beta, delta_log_Z, next_kl)

        def _cond_fn(carry):
            _, _, kl_increase = carry
            return kl_increase > detla_kl

        final_carry = lax.while_loop(_cond_fn, _body_fn, init_carry)

        final_delta_beta, final_delta_log_Z, _ = final_carry

        next_beta = current_beta + final_delta_beta
        next_log_Z = final_delta_log_Z + current_log_Z

        return next_beta, next_log_Z

    return adaptive_tempering_fn, init_log_Z
