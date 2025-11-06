"""Coarse-grained alanine dipeptide target from
https://doi.org/10.1021/acs.jctc.5c01504."""
from jax import jit, tree_util, numpy as jnp

import numpy as np
import equinox as eqx

from jax_md_mod import io
from jax_md_mod.custom_quantity import angular_displacement, _bond_length, \
                                       dihedral_displacement
from jax_md_mod.model import layers, neural_networks
from jax_md import partition, space

from chemtrain import quantity
import mdtraj
import haiku as hk

from ecg.targets.base import Target
from ecg.distributions import JointModelTransformed
from ecg.chemtrain_mod.custom_prior_new import ForceField, Topology
from ecg.chemtrain_mod.custom_prior import select_protein, select_priors, \
                                           prior_potential
from ecg.utils import bond_and_angle_names, init_map_x_to_r, load_beta_array
from ecg.visualization import plot_marginals, plot_matrix, \
                              plot_rama, plot_1d_dihedrals

from typing import Callable
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

DisplacementFn = space.DisplacementFn


FORCE_FIELD_PATH = 'data/alanine_heavy.toml'
TOPOLOGY_PATH = 'data/ala2_cg_heavy_2_7nm.gro'
RE_PARAMS_PATH = '../pretrained_models/relative_entropy_dimenet_aldp/'+\
                    'pretrained_params_re_aldp.pkl'


class AlanineDipeptide(Target):
    #TODO: more info in docstring
    """
    Alanine dipeptide example.
    """
    displacement_fn: DisplacementFn
    map_x_to_r: Callable[[ArrayLike], Array]
    _energy_fn: Callable[[ArrayLike], Array]
    fix_coord_idxs: int | Array | tuple
    dihedral_idxs: Array
    kT: Float
    topology: Topology

    def __init__(self,
                 temperature,
                 fix_coord_idxs,
                 ):

        self.kT = temperature * quantity.kb
        self.displacement_fn = space.free()[0]
        self.fix_coord_idxs = fix_coord_idxs

        energy_fn, map_x_to_r, topology = init_dimenet_model(
            self.displacement_fn, self.fix_coord_idxs, self.kT
        )

        self._energy_fn = energy_fn
        self.map_x_to_r = map_x_to_r
        self.topology = topology

        # 0: phi    1: psi
        self.dihedral_idxs = jnp.array([[1, 3, 4, 6], [3, 4, 6, 8]],
                                       dtype=jnp.int32)

    @eqx.filter_jit
    def energy(self, x: ArrayLike) -> Array:
        r = self.map_x_to_r(x)
        return self._energy_fn(r) / self.kT

    def _angles(self, x: ArrayLike) -> Array:
        r = self.map_x_to_r(x)
        angle_idxs, _, mask = self.topology.get_angles()
        angles = angular_displacement(r,
                                      self.displacement_fn,
                                      angle_idxs[mask])
        return angles

    def angles(self, x: ArrayLike) -> Array:
        """
        Compute bond angles for alanine dipeptide.

        Args:
            x: Points at which to evaluate bond angles.
        Returns:
            angles: Jax array of shape (..., num_angles) representing bond
                    angles in degrees.
        """
        return jnp.vectorize(self._angles, signature='(n)->(m)')(x)

    def bonds(self, x: ArrayLike) -> Array:
        """
        Compute bond lengths for alanine dipeptide.

        Args:
            x: Points at which to evaluate bond lengths.
        Returns:
            distances: Jax array of shape (..., num_bonds) representing bond
                       lengths.
        """
        r = jnp.vectorize(self.map_x_to_r, signature='(n)->(m,3)')(x)
        bond_idxs, _, mask = self.topology.get_bonds()
        distances = _bond_length(bond_idxs[mask], r, self.displacement_fn)
        return distances

    def _dihedral_angles(self, x: ArrayLike) -> Array:
        r = self.map_x_to_r(x)
        dihedrals = dihedral_displacement(
            r, self.displacement_fn, self.dihedral_idxs
        )
        return dihedrals

    def dihedral_angles(self, x: ArrayLike) -> Array:
        """
        Compute the dihedral angles (in degrees) for alanine dipeptide.

        Args:
            x: Points at which to evaluate dihedral angles.
        Returns:
            dihedrals: Jax array of shape (..., 2) representing the phi and
                       psi dihedral angles.
        """
        return jnp.vectorize(self._dihedral_angles, signature='(n)->(m)')(x)

    def ref_samples(self, beta: Float) -> Array:
        return load_beta_array(beta, return_beta=False)

    def evaluate(self,
                 model: JointModelTransformed,
                 key: PRNGKeyArray,
                 beta: Float,
                 num_samples: int = 10000,
                 save_name: str | None = None,
                 folder_name: str | None = None,
                 bins: int = 60,
                 ):
        #TODO: release: add evaluation of radius of gyration,
        # RMSD, energy, bond, and diversity scores

        samples = model.sample(key, (num_samples,))
        ref_samples = self.ref_samples(beta)

        pred_bonds = self.bonds(samples)
        ref_bonds = self.bonds(ref_samples)

        pred_angles = self.angles(samples)
        ref_angles = self.angles(ref_samples)

        bond_names, angle_names = bond_and_angle_names(self.topology)

        plot_marginals(pred_bonds,
                       ref_bonds,
                       bins=bins,
                       titles=bond_names,
                       single_axis_labels=['Density',
                                           r'Bond length in $\mathrm{nm}$'],
                       save_name=save_name+'bonds_' if save_name else None,
                       folder_name=folder_name and folder_name+ \
                                    'observables/')

        plot_marginals(pred_angles,
                       ref_angles,
                       bins=bins,
                       titles=angle_names,
                       single_axis_labels=['Density',
                                           r'Angle in $\mathrm{deg}$'],
                       save_name=save_name+'angles_' if save_name else None,
                       folder_name=folder_name and folder_name+ \
                                    'observables/')

    def plot(self,
             model: JointModelTransformed,
             key: PRNGKeyArray,
             beta: Float,
             num_samples: int = 100000,
             save_name: str | None = None,
             folder_name: str | None = None,
             bins: int = 60,
             annotate_matrix: bool = False,
             ):

        samples = model.sample(key, (num_samples,))
        dihedral_angles = self.dihedral_angles(samples)

        ref_samples = self.ref_samples(beta)
        ref_dihedral_angles = self.dihedral_angles(ref_samples)

        plot_marginals(samples,
                       ref_samples,
                       bins=bins,
                       axis_label_pattern=r'$x_{z_{%d}}$',
                       save_name=save_name,
                       folder_name=folder_name and folder_name+ \
                                    'marginals/')

        plot_rama([ref_dihedral_angles, dihedral_angles],
                  titles=['Predicted', 'Reference'],
                  save_name=save_name,
                  folder_name=folder_name and folder_name+ \
                               'dihedrals/')

        plot_1d_dihedrals([ref_dihedral_angles, dihedral_angles],
                          save_name=save_name,
                          folder_name=folder_name and folder_name+ \
                               'dihedrals/')
        matrix = model.matrix

        plot_matrix(matrix,
                    annot=annotate_matrix,
                    titles=['Transformation Matrix'],
                    fontsize=12,
                    labelsize=9,
                    annot_size=12,
                    save_name=save_name,
                    folder_name=folder_name and folder_name+ \
                                    'transformations/')


def init_dimenet_model(displacement_fn: DisplacementFn,
                       fix_coord_idxs: int | Array | tuple,
                       kT: float) -> tuple[Callable, Callable, Topology]:
    """Initialize the Dimenet model for alanine dipeptide.
    
    Args:
        displacement_fn: `jax_md` displacement function.
        fix_coord_idxs: Indices (Integer, tuple, or an ndarray with integer
            dtype) to remove from the flattened atomistic vector `r` to fix
            rigid body motions.
        
    Returns:
        energy_fn: The energy function.
        map_x_to_r: Function to map reduced coordinates to atomistic positions.
        topology: The molecular topology.
    """
    box, r_init, _, _ = io.load_box(TOPOLOGY_PATH)

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        r_cutoff=0.9,
        dr_threshold=0.05,
        disable_cell_list=True,
        capacity_multiplier=1.5
    )

    nbrs_init = neighbor_fn.allocate(r_init, extra_capacity=0)

    force_field = ForceField.load_ff(FORCE_FIELD_PATH)

    top = mdtraj.load_topology(TOPOLOGY_PATH)

    _mapping = force_field.mapping(by_name=True)

    def mapping(name='', residue='', **kwargs):
        if residue == 'NME' and name == 'C':
            return _mapping(name='CH3', **kwargs)
        if name == 'CB':
            return _mapping(name='CH3', **kwargs)
        else:
            return _mapping(name=name, **kwargs)

    topology = Topology.from_mdtraj(top, mapping,
                                        include_impropers=True,
                                        )

    priors = ['bond', 'angle', 'dihedral', 'improper']
    species, prior_idxs, prior_constants = select_protein(
                                    'heavy_alanine_dipeptide', priors)
    prior_energy_fn = select_priors(displacement_fn,
                                    prior_constants,
                                    prior_idxs,
                                    kT)
    n_species = species.shape[0]

    mlp_init = {
        'b_init': hk.initializers.Constant(0.),
        'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
    }


    _, gnn_energy_fn = neural_networks.dimenetpp_neighborlist(
        displacement_fn, 0.5, n_species, r_init, nbrs_init,
        embed_size=32, init_kwargs=mlp_init,
    )

    def energy_fn_template(energy_params):

        def _energy_fn(pos, **dynamic_kwargs):
            gnn_energy = gnn_energy_fn(
                energy_params, pos, neighbor=nbrs_init, species=species,
                **dynamic_kwargs
            )

            prior_energy = prior_potential(prior_energy_fn, pos,
                                           neighbor=nbrs_init,
                                           **dynamic_kwargs)

            return gnn_energy + prior_energy
        return _energy_fn

    re_params = tree_util.tree_map(
        jnp.asarray, np.load(RE_PARAMS_PATH, allow_pickle=True)
    )

    energy_fn = jit(energy_fn_template(re_params))
    map_x_to_r = init_map_x_to_r(r_init, fix_coord_idxs)

    return energy_fn, map_x_to_r, topology
