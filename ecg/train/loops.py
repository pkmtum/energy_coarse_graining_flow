"""Training loops."""
from collections.abc import Callable
from tqdm import tqdm

from jax import random

import equinox as eqx
import paramax
import optax

from ecg.train.train_utils import TrainState, init_tempering, step_aux

from typing import Any, Generator
from jaxtyping import Float, PyTree, PRNGKeyArray, Scalar


def fit_to_key_beta_based_loss(
    key: PRNGKeyArray,
    tree: PyTree,
    beta: Float,
    *,
    loss_fn: Callable[[PyTree, PyTree, PRNGKeyArray, float], Scalar],
    steps: int,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    opt_state: PyTree | None,
    return_best: bool = False,
    show_progress: bool = True,
    return_bij_params: bool = False,
):
    """Train a pytree, using a loss with params, static, key, and beta 
    as arguments.

    This is can be used to train :math:`KL(q||p;\beta)` with temperature 
    dependence, e.g., https://arxiv.org/pdf/2504.20940.

    Args:
        key: Jax random key.
        tree: PyTree, from which trainable parameters are found using
            ``equinox.is_inexact_array``.
        beta: Inverse temperature.
        loss_fn: The loss function to optimize.
        steps: The number of optimization steps.
        learning_rate: The adam learning rate. Ignored if optimizer is provided.
        optimizer: Optax optimizer. Defaults to None.
        return_best: Whether the result should use the parameters where the minimum 
            loss was reached (when True), or the parameters after the last update 
            (when False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained pytree and the losses.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        tree,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )

    best_params = params

    if opt_state is None:
        opt_state = optimizer.init(params)

    losses = []
    grad_norms = []

    if return_bij_params:
        bij_params = []

    keys = tqdm(random.split(key, steps), disable=not show_progress)

    for key in keys:
        params, opt_state, loss, grad_norm = step_aux(
            params,
            static,
            key=key,
            beta=beta,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
        )
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())
        keys.set_postfix({'loss': loss.item(),
                          'beta': f'{beta:.5f}'})

        if losses[-1] == min(losses):
            best_params = params

        if return_bij_params:
            bij_params.append(params.bijection.bijections[0].matrix)

    params = best_params if return_best else params

    infos = TrainState(opt_state=opt_state,
                       losses=losses,
                       grad_norm=grad_norms,
                       bij_params=bij_params if return_bij_params else [])
    return eqx.combine(params, static), infos


class TemperingScheme:
    """Class representing a tempering scheme for controlling 
        inverse temperature (beta) from https://arxiv.org/pdf/2504.20940.

        Args:
            tempering_fn: The function that updates beta and state for each 
                          tempering step.
            beta: Current inverse temperature (beta).
            target_beta: Target inverse temperature.
            state: State with additional information for tempering.
            method: 'linear', 'adaptive', 'array', or None.
            kwargs: Extra parameters (delta_beta, delta_kl, num_samples, array,
            get_log_weights, ...).
        """
    tempering_fn: Callable[
        [PyTree, PRNGKeyArray, Float, Any],
        tuple[Scalar, Any],
                  ]
    beta: Float
    target_beta: Float
    state: Any
    method: str | None
    kwargs: dict
    initialized: bool = False
    converged: int = 0

    def __init__(self,
                 start_beta: Float,
                 target_beta: Float,
                 **kwargs):

        self.kwargs = kwargs
        self.method = self.kwargs['method']
        self.beta = start_beta
        self.target_beta = target_beta

        assert self.beta <= self.target_beta, (
        f'Initial beta has to be <= final beta, '
        f'but got start_beta={self.beta}, target_beta={self.target_beta}'
            )


    def _initialize(self, model: PyTree, key: PRNGKeyArray):
        """Initialize tempering function inside step."""
        self.tempering_fn, self.state = init_tempering(model,
                                                       key,
                                                       self.beta,
                                                       **self.kwargs)
        self.initialized = True

    def step(self, model: PyTree, key: PRNGKeyArray) -> float:
        """One step in the tempering scheme."""
        if not self.initialized:
            key, init_key = random.split(key)
            self._initialize(model, init_key)

        new_beta, state = self.tempering_fn(model,
                                            key,
                                            self.beta,
                                            self.state)

        self.state = state
        self.beta = min(new_beta, self.target_beta)
        return self.beta

    def loop(self) -> Generator[None, None, None]:
        """
        Generator for training loop.
        - Yields control for training at the current beta.
        - Guarantees training at start_beta first.
        - Guarantees training at target_beta last, even if the 
          last step overshoots.
        """
        while True:
            yield
            if self.beta >= self.target_beta:
                self.converged += 1
            if self.converged >= 2:
                break
