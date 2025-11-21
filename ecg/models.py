"""Module for initializing model components."""
from jax import random, numpy as jnp

import numpy as np

import paramax
from flowjax import flows
from flowjax.bijections import Chain, Exp, Indexed, Permute, \
                               RationalQuadraticSpline

from ecg.bijections import SignFlip, StochasticLinearBijection, \
                           StochasticLinearBijectionPerAtom
from ecg.distributions import JointDistribution, JointModelTransformed, \
                              MLPConditionalNormal, TruncatedNormal
from ecg.utils import Dimensions

from jaxtyping import Array, Int, PRNGKeyArray


def initialize_model(dimensions: Dimensions,
                     Phi_init: Array,
                     key: PRNGKeyArray = random.key(0),
                     flow_layers: int = 6,
                     nn_depth: int = 2,
                     nn_width: int = 32,
                     knots: int = 8,
                     interval: float = 5,
                     cond_nn_depth: int = 2,
                     cond_nn_width: int = 32,
                     perm_idxs: Int[Array | np.ndarray, '...'] | None = None,
                     restrain_idxs: int | Array | tuple | None = None,
                     per_atom: bool = False,
                     load_params_path: str | None = None,
                     ) -> JointModelTransformed:
    """Initialize model components for flow, conditional distribution, and
    linear transformation, and return combined model.

    Args:
        dimensions (Dimensions): Object containing coare-grained (`z_dim`) 
            and atomistic (`x_dim`) dimensions.
        A_init (Array): Initial linear transformation matrix for 
            `StochasticLinearBijection`.
        key (PRNGKeyArray, optional): Random key for initialization. 
            Defaults to `random.key(0)`.
        load_params_path (str, optional): Path to load pre-trained model 
            parameters. Defaults to None.
        flow_layers (int, optional): Number of flow layers. Defaults to 6.
        nn_width (int, optional): Width of neural networks used in the flow 
            and conditional distributions. Defaults to 32.
        nn_depth (int, optional): Depth of neural networks used in the flow 
            and conditional distributions. Defaults to 2.
        knots (int, optional): Number of knots in rational quadratic spline 
            transformers. Defaults to 8.
        interval (float, optional): Range for truncated normal base 
            distribution and spline transformation. Defaults to 5.
        cond_nn_depth (int, optional): Depth of neural networks used in the 
            conditional distribution. Defaults to 2.
        cond_nn_width (int, optional): Width of neural networks used in the
            conditional distribution. Defaults to 32.
        perm_idxs (Int[Array | np.ndarray, "..."], optional): Permutation 
            indices atom ordering. If None, no permutation is applied. 
            Defaults to None.
        restrain_idxs (int | Array | tuple, optional): Indices to apply 
            exponential and sign flip restraint to keep coordinates negative.
            If None, no restraint is applied. Defaults to None.

    Returns:
        model (JointModelTransformed): Transformed distribution of the joint 
                                       model.
    """
    flow_key, cond_key = random.split(key)

    base_dist = paramax.non_trainable(TruncatedNormal(
                                jnp.full(dimensions.z_dim, -interval),
                                jnp.full(dimensions.z_dim, interval)))

    if dimensions.z_dim == 1:
        # No coupling flow for one dimension
        flow = flows.masked_autoregressive_flow(
                        flow_key, base_dist=base_dist, flow_layers=flow_layers,
                        nn_width=nn_width, nn_depth=nn_depth,
                        transformer=RationalQuadraticSpline(knots=knots,
                                                            interval=interval))
    else:
        flow = flows.coupling_flow(
                        flow_key, base_dist=base_dist, nn_width=nn_width,
                        nn_depth=nn_depth, flow_layers=flow_layers,
                        transformer=RationalQuadraticSpline(knots=knots,
                                                        interval=interval))

    if per_atom:
        #TODO: calculate remove row ids from fix_coords_idxs
        remove_idxs = jnp.array([6,8,10], dtype=jnp.int32)
        transformation = StochasticLinearBijectionPerAtom(
            init=Phi_init, remove_rows_idxs=remove_idxs)
    else:
        transformation = StochasticLinearBijection(init=Phi_init)
    bijections = [transformation]

    cond_dist = MLPConditionalNormal(cond_key,
                                     dimensions.X_dim,
                                     dimensions.z_dim,
                                     nn_width=cond_nn_width,
                                     nn_depth=cond_nn_depth)

    joint_dist = JointDistribution(flow, cond_dist)

    if perm_idxs is not None:
        assert perm_idxs.shape[0] == dimensions.x_dim, (
            f'Permutation indices shape {perm_idxs.shape} must match'
            f'x_dim shape {dimensions.x_dim}.')
        permutation = Permute(perm_idxs)
        bijections.append(paramax.non_trainable(permutation))

    if restrain_idxs is not None:
        indexed_shape = jnp.asarray(restrain_idxs).shape
        indexed_bijection = Indexed(Chain([Exp(indexed_shape),
                                        SignFlip(indexed_shape)]),
                                    idxs=restrain_idxs,
                                    shape=(dimensions.x_dim,))
        bijections.append(paramax.non_trainable(indexed_bijection))

    model = JointModelTransformed(base_dist=joint_dist,
                                  bijection=Chain(bijections))

    model = model.load(load_params_path)

    return model
