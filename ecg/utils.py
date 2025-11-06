"""Utility functions."""
import os
import re
import dataclasses

from pathlib import Path

import jax
import jax.numpy as jnp

import numpy as np

from flowjax.utils import arraylike_to_array

import optax

from ecg.train.optimizer import OptimizerConfig, get_optimizer
from ecg.chemtrain_mod.custom_prior_new import Topology

from typing import Callable
from jaxtyping import Array, ArrayLike, Float


@dataclasses.dataclass
class Dimensions:
    """A dataclass containing dimension information.

    Attributes:
        x_dim: Dimension of atomistic coordinates.
        z_dim: Dimension of coarse-grained coordinates.
        num_dofs: Number of removed degrees of freedom to fix roto-translational
            invariance. Defaults to None.
    """

    x_dim: int
    z_dim: int
    num_dofs: int = 0

    def __init__(self, x_dim: int,
                 z_dim: int,
                 dofs: int | Array | tuple | None = None):
        self.x_dim = x_dim
        self.z_dim = z_dim
        num_dofs = jnp.atleast_1d(dofs).shape[0] if dofs is not None else 0
        self.num_dofs = num_dofs

    @property
    def X_dim(self) -> int:
        return self.x_dim - self.z_dim

    @property
    def num_atoms(self) -> int:
        return int((self.x_dim + self.num_dofs) / 3)


# https://github.com/google/jax/issues/2680
def fill_diagonal(a, val):
    """Fills the diagonal of a matrix with a value."""
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


def kron_block(a, identity):
    """3x9 block whose rows are a⊗e0, a⊗e1, a⊗e2."""
    return jax.vmap(lambda e: jnp.kron(a, e))(identity)


def ab_kl_div(a, b, zX, joint_log_prob, target, beta=1.0):
    """Calculate the reverse KL divergence KL(q || p) for 
    a given matrix."""

    matrix = jnp.array([
        [a, 1.0 - a],
        [b, 1.0 - b]
    ])

    x = jax.vmap(jnp.matmul, (None,0))(matrix, zX)
    x_log_dets = jnp.linalg.slogdet(matrix)[1]

    beta_phi_potentials = beta * jax.vmap(target.energy)(x) - x_log_dets

    return beta_phi_potentials.mean() + joint_log_prob + target.log_Z(beta)


def init_map_x_to_r(positions: ArrayLike,
                    fix_coord_idxs: int | Array | tuple,
                    zero_dof: bool = True,
                    ) -> Callable[[Array], Array]:
    """Create a function to reconstruct full atomistic positions from a
    reduced coordinate vector by reinserting removed degrees of freedom.

    Args:
        positions: Full-atom positions used to extract the fixed coordinates.
            Accepts an array of shape (N, 3).
        fix_coord_idxs: Indices (Integer, tuple, or an ndarray with integer
            dtype) to remove from the flattened atomistic vector `r` to fix
            rigid body motions.
        zero_dof: If True, the fixed coordinates are set to zero instead of
            their original values in `positions`.

    Returns:
        map_x_to_r: A function f(x) that takes a flattened reduced
            coordinate vector `x`, inserts the fixed coordinates at the
            original locations, and returns the reconstructed atomistic
            positions shaped (N, 3).
    """

    fix_coord_idxs = jnp.asarray(fix_coord_idxs)
    fix_coord_idxs = jnp.atleast_1d(fix_coord_idxs)
    positions = arraylike_to_array(positions)
    if zero_dof:
        positions = jnp.zeros(positions.shape)
    fixed_positions = positions.flatten().at[fix_coord_idxs].get()

    num_removed_before_each = jnp.arange(len(fix_coord_idxs))
    fixed_idx = fix_coord_idxs - num_removed_before_each

    @jax.jit
    def map_x_to_r(x):
        assert x.shape == (positions.size - fixed_idx.size,), (
            'Input x must have shape '
            f'{(positions.size - fixed_idx.size,)}, got {x.shape}'
        )
        atomisitic_coordinates = jnp.insert(x, fixed_idx, fixed_positions)
        return atomisitic_coordinates.reshape((-1,3))

    return map_x_to_r


def init_optimizer(learning_rate: float,
                   dynamic_grad_ignore_factor: float = 10.,
                   dynamic_grad_norm_factor: float = 5., #2.
                   dynamic_grad_norm_window: int = 50, #25
    ) -> optax.GradientTransformation | optax.GradientTransformationExtraArgs:
    """Initialize the custom optimizer from SE3 Augmented Coupling Flows. 
    https://github.com/lollcat/se3-augmented-coupling-flows"""

    optimizer_config = OptimizerConfig(
                init_lr= learning_rate,
                optimizer_name = 'adam',
                dynamic_grad_ignore_and_clip = True,
                dynamic_grad_ignore_factor = dynamic_grad_ignore_factor,
                dynamic_grad_norm_factor  = dynamic_grad_norm_factor,
                dynamic_grad_norm_window= dynamic_grad_norm_window,
                )

    optimizer = get_optimizer(optimizer_config)[0]
    return optimizer


def load_beta_array(beta: Float,
                    base_path: str = '../datasets/aldp/',
                    return_beta: bool = False) -> Array | tuple[Array, Float]:
    """
    Load the numpy array for the given beta, or the next higher available beta.

    Args:
        beta: The beta value to load.
        base_path: Directory containing the beta files.
        return_beta: If True, also return the actual beta value of the loaded 
                     file.

    Returns:
        Numpy array corresponding to the requested or next-highest beta.
        If `return_beta` is True, also returns the actual beta value.
    """

    pattern = re.compile(r'beta([0-9]*\.?[0-9]+)_aldp_samples\.npy')

    beta_files = []
    for name in os.listdir(base_path):
        match = pattern.match(name)
        if match:
            beta_value = float(match.group(1))
            beta_files.append((beta_value, os.path.join(base_path, name)))

    if not beta_files:
        raise FileNotFoundError(f'No beta files found in {base_path}')

    beta_files.sort(key=lambda x: x[0])

    # Find the next-highest beta
    for b_val, path in beta_files:
        if b_val >= beta:
            return (np.load(path), b_val) if return_beta else np.load(path)

    # If no higher beta, return the largest available
    b_val, path = beta_files[-1]
    return (np.load(path), b_val) if return_beta else np.load(path)


def create_output_dirs(folder_name: str | None,
                       save_params_path: str | None,
                       subdirs: list[str] | None = None,
                       ):
    """
    Create output folders for figures and trained parameters.

    Args:
        folder_name (str): Name of the folder inside outputs/.
        save_params_path (str | None): If not None, also create 
        trained_parameter folders.
        subdirs (list[str] | None): List of subdirectories to create under 
        outputs/figures/folder_name.
    """

    if folder_name is None:
        return

    base_fig_dir = Path(f'outputs/figures/{folder_name}')
    base_fig_dir.mkdir(parents=True, exist_ok=True)

    if subdirs:
        for sub in subdirs:
            (base_fig_dir / sub).mkdir(parents=True, exist_ok=True)

    if save_params_path is not None:
        base_param_dir = Path(f'outputs/trained_parameter/{folder_name}')
        base_param_dir.mkdir(parents=True, exist_ok=True)
        (base_param_dir / 'intermediates').mkdir(parents=True, exist_ok=True)


def bond_and_angle_names(topology: Topology) -> tuple[list[str], list[str]]:
    """
    topology: Custom chemtrain topology object.
    
    Returns:
        bond_names: list of strings like 'H-C'
        angle_names: list of strings like 'H-C-O'
    """

    atom_names = topology._names
    atom_names[len(atom_names)-1] = 'CH3'

    bond_idxs, _, bond_mask = topology.get_bonds()
    angle_idxs, _, angle_mask = topology.get_angles()

    atom_names = [
        fr'${atom_names[i]}^{{({(i+1)})}}$'
        for i in range(len(atom_names))
    ]

    bond_names = [
        fr'{atom_names[i]} — {atom_names[j]}'
        for i, j in bond_idxs[bond_mask]
    ]

    angle_names = [
        f'{atom_names[i]} — {atom_names[j]} — {atom_names[k]}'
        for i, j, k in angle_idxs[angle_mask]
    ]

    return bond_names, angle_names
