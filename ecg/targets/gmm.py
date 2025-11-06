"""GMM target from https://doi.org/10.1021/acs.jctc.5c01504."""
from collections.abc import Iterable

from jax import random, numpy as jnp

import numpy as np
import equinox as eqx
from flowjax.bijections import Chain, Scale, AdditiveCondition
from flowjax.distributions import AbstractDistribution, \
                                  Transformed, StandardNormal

from sklearn.datasets import make_blobs

from ecg.targets.base import Target
from ecg.distributions import GaussianMixture, JointDistribution, \
                              JointModelTransformed
from ecg.visualization import plot_marginals, plot_matrix, \
                              plot_contour_and_samples

from jaxtyping import Array, ArrayLike, PRNGKeyArray


class GMM(Target):
    r"""
    Target distribution based on Gaussian mixture model (GMM).

    This synthetic target is constructed as a three-component GMM 
    over a subset of the coordinates, with a conditional Gaussian 
    over the remaining coordinates. Specifically, the all-atom coordinates 
    :math:`x` are partitioned as :math:`(x_X, x_z)`, and the density is 
    defined as

    .. math::

        p(x_X, x_z) = p(x_X \mid x_z) \, p(x_z),

    where the marginal over :math:`x_z` is a mixture of three Gaussians:

    .. math::

        p(x_z) = \sum_{k=1}^{3} w_k \, \mathcal{N}(x_z \mid m_k, \Sigma_k),

    with equal weights :math:`w_k = 1/3`, means :math:`m_k` sampled uniformly 
    from :math:`[-1, 1]^{\mathrm{dim}(z)}`, and diagonal covariances 
    :math:`\Sigma_k = \mathrm{diag}(0.01)`.

    The conditional over :math:`x_X` is a diagonal Gaussian with a linear 
    dependence on :math:`x_z`:

    .. math::

        p(x_X \mid x_z) = \mathcal{N}(x_X \mid B x_z, S),

    where the entries of the matrix :math:`B` are sampled from a standard 
    normal distribution and the covariance :math:`S = \mathrm{diag}(0.01)`.
    """
    joint_dist: JointDistribution
    gmm: AbstractDistribution
    cond_dist: AbstractDistribution
    cluster_components: int
    cluster_std: float
    seed: int

    def __init__(self,
                 seeds,
                 dimensions,
                 cluster_components,
                 cluster_std,
                 cond_std):

        self.cluster_components = cluster_components
        self.cluster_std = cluster_std
        self.seed = seeds['gmm']

        params = make_blobs_with_params_wrapper(
                             n_features=dimensions.z_dim,
                             center_box=(-1,1),
                             cluster_std=self.cluster_std,
                             centers=self.cluster_components,
                             n_samples=1,
                             random_state=self.seed,
                             )


        means, covs, weights = params
        self.gmm = GaussianMixture(means, covs, weights)

        key = random.key(seeds['cond'])
        B = random.normal(key, (dimensions.X_dim, dimensions.z_dim))
        cond_bijections = Chain([
                        Scale(jnp.full(dimensions.X_dim, cond_std)),
                        AdditiveCondition(lambda z: B @ z,
                                          shape=(dimensions.X_dim,),
                                          cond_shape=(dimensions.z_dim,)),
                        ])
        self.cond_dist = Transformed(StandardNormal((dimensions.X_dim,)),
                                cond_bijections)

        self.joint_dist = JointDistribution(self.gmm, self.cond_dist)

    @eqx.filter_jit
    def energy(self, x: ArrayLike) -> Array:
        return - self.joint_dist.log_prob(x)

    def gmm_params(self):
        return self.gmm.means, self.gmm.covariances, self.gmm.weights

    def ref_samples(self, num_samples: int = 10000) -> Array:
        gmm_samples, _ = make_blobs_with_params_wrapper(
                             n_features=self.gmm.shape[0],
                             center_box=(-1,1),
                             cluster_std=self.cluster_std,
                             centers=self.cluster_components,
                             n_samples=num_samples,
                             random_state=self.seed,
                             return_samples=True,
                             )
        return gmm_samples

    def plot(self,
             model: JointModelTransformed,
             key: PRNGKeyArray,
             num_samples: int = 10000,
             save_name: str | None = None,
             folder_name: str | None = None,
             dims: list[int] | None = None,
             bins: int = 50,
             annotate_matrix: bool = False,
             ):
        samples = model.sample_and_split(key, (num_samples,))[0]
        ref_samples = self.ref_samples(num_samples=num_samples)

        plot_contour_and_samples(self,
                                dims=dims,
                                samples=samples,
                                save_name=save_name,
                                folder_name=folder_name and folder_name+ \
                                        'contours/')

        plot_marginals(samples,
                       ref_samples,
                       axis_label_pattern=r'$x_{z_{%d}}$',
                       bins=bins,
                       save_name=save_name,
                       folder_name=folder_name and folder_name+ \
                                    'marginals/')

        x_dim = model.shape[0]
        z_dim = model.base_dist.z_dim
        z_ticks = [fr'$z_{{{i}}}$' for i in range(z_dim)]
        X_ticks = [fr'$X_{{{i}}}$' for i in range(x_dim-z_dim)]
        x_z_ticks = [fr'$x_{{z_{{{i}}}}}$' for i in range(z_dim)]
        x_X_ticks = [fr'$x_{{X_{{{i}}}}}$' for i in range(x_dim-z_dim)]
        x_tick = x_z_ticks + x_X_ticks

        matrix = model.matrix

        plot_matrix(matrix,
                    annot=annotate_matrix,
                    titles=['Transformation Matrix'],
                    yticklabels=x_tick,
                    xticklabels=z_ticks+X_ticks,
                    fontsize=12,
                    labelsize=9,
                    annot_size=12,
                    save_name=save_name,
                    folder_name=folder_name and folder_name+ \
                                    'transformations/')

        inv_matrix = jnp.linalg.inv(matrix)

        plot_matrix(inv_matrix,
                    annot=annotate_matrix,
                    titles=['Inverse Transformation Matrix'],
                    xticklabels=x_tick,
                    yticklabels=z_ticks+X_ticks,
                    fontsize=12,
                    labelsize=9,
                    annot_size=12,
                    save_name=save_name+'_inv' if save_name else None,
                    folder_name=folder_name and folder_name+ \
                                    'transformations/')


def make_blobs_with_params_wrapper(
    *args,
    return_samples=False,
    random_state=None,
    **kwargs,
    ) -> tuple[Array, Array, Array] | tuple[Array, Array]:
    """Wrapper around sklearn.datasets.make_blobs that adds weights, 
    stds, covariances of the blobs for a Gaussian Mixture Model."""
    X, y, centers = make_blobs(*args, return_centers=True,
                               random_state=random_state, **kwargs)

    n_features = X.shape[1]
    n_centers = centers.shape[0]
    cluster_std = kwargs.get('cluster_std', 1.0)

    if np.isscalar(cluster_std):
        cluster_std = np.full(n_centers, cluster_std, dtype=float)
    else:
        cluster_std = np.asarray(cluster_std, dtype=float)

    n_samples = kwargs.get('n_samples', None)

    if isinstance(n_samples, Iterable):
        n_samples_arr = np.asarray(n_samples, dtype=float)
        weights = n_samples_arr / n_samples_arr.sum()
    else:
        weights = np.full(n_centers, 1.0 / n_centers)

    covariances = jnp.stack([np.eye(n_features) * (s**2) for s in cluster_std])

    if return_samples:
        return  X, y
    else:
        return centers, covariances, weights
