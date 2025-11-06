"""Base target distribution class."""
from abc import abstractmethod

import equinox as eqx

from jaxtyping import Array, ArrayLike, PyTree, PRNGKeyArray


class Target(eqx.Module):
    """Abstraction target distribution class.
    This class defines the interface for target distributions.
    """

    @abstractmethod
    def energy(self, x: ArrayLike) -> Array:
        """Evaluate the energy function of point x."""

    @abstractmethod
    def ref_samples(self, **kwargs) -> Array:
        """Samples from the target distribution."""

    @abstractmethod
    def plot(self,
             model: PyTree,
             key: PRNGKeyArray,
             *args,
             num_samples: int = 10000,
             save_name: str | None,
             folder_name: str | None,
             **kwargs,
             ):
        """Creates the plots for the results in the paper."""
