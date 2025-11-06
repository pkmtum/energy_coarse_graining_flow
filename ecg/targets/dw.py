"""Double-well target from https://doi.org/10.1021/acs.jctc.5c01504."""
from jax import tree_util, random, numpy as jnp

import numpy as np
import equinox as eqx
from flowjax.utils import arraylike_to_array
from scipy.integrate import simpson

from ecg.targets.base import Target
from ecg.visualization import plot_energy_marginals, plot_2d_histogram, \
                             plot_kl_div

from jaxtyping import Array, ArrayLike, PyTree, PRNGKeyArray, Float


class DoubleWell(Target):
    r"""
    Double-well potential in the :math:`x_1` direction with harmonic
    confinement in the :math:`x_2` direction.

    The total energy is

    .. math::

        U(x_1, x_2) = a x_1^4 + b x_1^2 + c x_1 + d x_2^2.
    """
    a: float = 0.25
    b: float = -3.
    c: float = 1.
    d: float = 0.5

    @eqx.filter_jit
    def energy(self, x: ArrayLike) -> Array:
        x = arraylike_to_array(x, err_name='x', dtype=float)

        x_1, x_2 = jnp.split(x, 2, axis=-1)
        harmonic_confinement = self.energy_x2(x_2)
        double_well = self.energy_x1(x_1)
        return jnp.sum(harmonic_confinement + double_well, axis=-1)

    def energy_x1(self, x_1: jnp.ndarray) -> Array:
        double_well = self.a * x_1**4 +  self.b * x_1**2 + self.c * x_1
        return double_well

    def energy_x2(self, x_2: jnp.ndarray) -> Array:
        harmonic_confinement = self.d * x_2 ** 2
        return harmonic_confinement

    def _log_Z_x1(self, beta: Float=1.) -> Array:
        x1_vals = jnp.linspace(-5, 5, 1000)
        analytic_x1 = self.energy_x1(x1_vals) * beta
        partition_function_x1 = simpson(jnp.exp(-analytic_x1),
                                        x=x1_vals)
        return jnp.log(partition_function_x1)

    def _log_Z_x2(self, beta: Float=1.) -> Array:
        return 0.5 * jnp.log(jnp.pi*2/beta)

    def log_Z(self, beta: Float=1.) -> Array:
        return self._log_Z_x1(beta) + self._log_Z_x2(beta)

    def p_x(self, x: Array, beta: Float=1.) -> Array:
        return jnp.exp(-beta * self.energy(x) - self._log_Z(beta))

    def p_x1(self, x_1: Array, beta: Float=1.) -> Array:
        return jnp.exp(-beta * self.energy_x1(x_1) - self._log_Z_x1(beta))

    def p_x2(self, x_2: Array, beta: Float=1.) -> Array:
        return jnp.exp(-beta * self.energy_x2(x_2) - self._log_Z_x2(beta))

    def ref_samples(self) -> Array:
        return np.load(self.reference_path)

    @property
    def reference_path(self) -> str:
        return '../datasets/dw/dw_samples.npy'

    def plot(self,
             model: PyTree,
             key: PRNGKeyArray,
             beta: Float,
             num_samples: int = 100000,
             save_name: str | None = None,
             folder_name: str | None = None,
             bij_params: list[ArrayLike] | None = None,
             ):
        ab_list = tree_util.tree_map(lambda a: a[:,0], bij_params)

        kl_key, energy_key, histo_key = random.split(key, 3)

        plot_kl_div(model,
                    kl_key,
                    self,
                    beta=beta,
                    num_samples=1000,
                    trajectory=ab_list,
                    save_name=save_name,
                    folder_name=folder_name and folder_name+\
                                'kl/')

        plot_energy_marginals(model,
                            energy_key,
                            self,
                            beta=beta,
                            num_samples=num_samples,
                            save_name=save_name,
                            folder_name=folder_name and folder_name+ \
                                        'energy_marginals/')

        plot_2d_histogram(model,
                          histo_key,
                          self,
                          num_samples=num_samples,
                          save_name=save_name,
                          folder_name=folder_name and folder_name+\
                                    'histograms/')
        