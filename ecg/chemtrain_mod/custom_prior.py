"""Prior potentials for interatomic potentials from 
https://github.com/tummfm/relative-entropy."""
import jax.numpy as jnp

import numpy as np

from jax_md_mod import custom_energy
from jax_md import energy

from ecg.chemtrain_mod.custom_energy import harmonic_dihedral


def select_protein(protein, prior_list):
    idxs = {}
    constants = {}
    if protein == 'heavy_alanine_dipeptide':
        print('Distinguishing different C_Hx atoms')
        species = jnp.array([6, 1, 8, 7, 2, 6, 1, 8, 7, 6])
        if 'bond' in prior_list:
            bond_mean = np.load('data/prior/Alanine_dipeptide_heavy_eq_bond'
                                 '_length.npy')
            bond_variance = np.load('data/prior/Alanine_dipeptide_heavy_eq'
                                     '_bond_variance.npy')
            bond_idxs = np.array([[0, 1],
                                   [1, 2],
                                   [1, 3],
                                   [4, 6],
                                   [6, 7],
                                   [4, 5],
                                   [3, 4],
                                   [6, 8],
                                   [8, 9]])
            idxs['bond'] = bond_idxs
            constants['bond'] = (bond_mean, bond_variance)

        if 'angle' in prior_list:
            angle_mean = np.load('data/prior/Alanine_dipeptide_heavy_eq'
                                  '_angle.npy')
            angle_variance = np.load('data/prior/Alanine_dipeptide_heavy_eq'
                                      '_angle_variance.npy')
            angle_idxs = np.array([[0, 1, 2],
                                    [0, 1, 3],
                                    [2, 1, 3],
                                    [1, 3, 4],
                                    [3, 4, 5],
                                    [3, 4, 6],
                                    [5, 4, 6],
                                    [4, 6, 7],
                                    [4, 6, 8],
                                    [7, 6, 8],
                                    [6, 8, 9]])
            idxs['angle'] = angle_idxs
            constants['angle'] = (angle_mean, angle_variance)

        if 'LJ' in prior_list:
            lj_sigma = np.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            lj_epsilon = np.load('data/prior/Alanine_dipeptide_heavy_'
                                  'epsilon.npy')
            lj_idxs = np.array([[0, 5],
                                 [0, 6],
                                 [0, 7],
                                 [0, 8],
                                 [0, 9],
                                 [1, 7],
                                 [1, 8],
                                 [1, 9],
                                 [2, 5],
                                 [2, 6],
                                 [2, 7],
                                 [2, 8],
                                 [2, 9],
                                 [3, 9],
                                 [5, 9]])
            idxs['LJ'] = lj_idxs
            constants['LJ'] = (lj_sigma, lj_epsilon)

        if 'dihedral' in prior_list:
            dihedral_phase = np.load('data/prior/Alanine_dipeptide_heavy_'
                                      'dihedral_phase.npy')
            dihedral_constant = np.load('data/prior/Alanine_dipeptide_heavy'
                                         '_dihedral_constant.npy')
            dihedral_n = np.load('data/prior/Alanine_dipeptide_heavy_dihedral'
                                  '_multiplicity.npy')

            dihedral_idxs = np.array([[1, 3, 4, 6],
                                       [3, 4, 6, 8],
                                       [0, 1, 3, 4],
                                       [2, 1, 3, 4],
                                       [1, 3, 4, 5],
                                       [5, 4, 6, 8],
                                       [4, 6, 8, 9],
                                       [7, 6, 8, 9]])
            idxs['dihedral'] = dihedral_idxs
            constants['dihedral'] = (dihedral_phase, dihedral_constant,
                                     dihedral_n)

        if 'repulsive_nonbonded' in prior_list:
            # repulsive part of the LJ
            if 'LJ' in prior_list:
                raise ValueError('Not sensible to have LJ and repulsive part of'
                                 ' LJ together. Choose one.')
            ren_sigma = np.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            ren_epsilon = np.load('data/prior/Alanine_dipeptide'
                                   '_heavy_epsilon.npy')
            ren_idxs = np.array([[0, 5],
                                  [0, 6],
                                  [0, 7],
                                  [0, 8],
                                  [0, 9],
                                  [1, 7],
                                  [1, 8],
                                  [1, 9],
                                  [2, 5],
                                  [2, 6],
                                  [2, 7],
                                  [2, 8],
                                  [2, 9],
                                  [3, 9],
                                  [5, 9]])
            idxs['repulsive_nonbonded'] = ren_idxs
            constants['repulsive_nonbonded'] = (ren_sigma, ren_epsilon)

        if 'improper' in prior_list:
            improper_mean = np.load('data/prior/Alanine_dipeptide_heavy_eq'
                                  '_improper.npy')
            improper_variance = np.load('data/prior/Alanine_dipeptide_heavy_eq'
                                      '_improper_variance.npy')
            improper_idxs = np.array([[3, 4, 6, 5]], dtype=jnp.int32)

            idxs['improper'] = improper_idxs
            constants['improper'] = (improper_mean, improper_variance)
    else:
        raise ValueError(f'The protein {protein} is not implemented.')
    return species, idxs, constants


def select_priors(displacement, prior_constants, prior_idxs, kbt=None):
    """Build prior potential from combination of classical potentials."""
    prior_fns = {}
    if 'bond' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for bond prior.'
        bond_mean, bond_variance = prior_constants['bond']
        bonds = prior_idxs['bond']
        prior_fns['bond'] = energy.simple_spring_bond(
            displacement, bonds, length=bond_mean, epsilon=kbt / bond_variance)

    if 'angle' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for angle prior.'
        angle_mean, angle_variance = prior_constants['angle']
        angles = prior_idxs['angle']
        prior_fns['angle'] = custom_energy.harmonic_angle(
            displacement, angles, angle_mean, angle_variance, kbt)

    if 'LJ' in prior_constants:
        lj_sigma, lj_epsilon = prior_constants['LJ']
        lj_idxs = prior_idxs['LJ']
        prior_fns['LJ'] = custom_energy.lennard_jones_nonbond(
            displacement, lj_idxs, lj_sigma, lj_epsilon)

    if 'repulsive' in prior_constants:
        re_sigma, re_epsilon, re_cut, re_exp = prior_constants['repulsive']
        prior_fns['repulsive'] = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=re_sigma, epsilon=re_epsilon, exp=re_exp,
            initialize_neighbor_list=False, r_onset=0.9 * re_cut,
            r_cutoff=re_cut)

    if 'dihedral' in prior_constants:
        dih_phase, dih_constant, dih_n = prior_constants['dihedral']
        dihdral_idxs = prior_idxs['dihedral']
        prior_fns['dihedral'] = custom_energy.periodic_dihedral(
            displacement, dihdral_idxs, dih_phase, dih_constant, dih_n)

    if 'repulsive_nonbonded' in prior_constants:
        # only repulsive part of LJ via idxs instead of nbrs list
        ren_sigma, ren_epsilon = prior_constants['repulsive_nonbonded']
        ren_idxs = prior_idxs['repulsive_nonbonded']
        prior_fns['repulsive_1_4'] = custom_energy.generic_repulsion_nonbond(
            displacement, ren_idxs, sigma=ren_sigma, epsilon=ren_epsilon, exp=6)

    if 'improper' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for angle prior.'
        improper_mean, improper_variance = prior_constants['improper']
        improper_idx = prior_idxs['improper']
        prior_fns['improper'] = harmonic_dihedral(
            displacement, improper_idx, improper_mean, improper_variance, kbt)
    return prior_fns


def prior_potential(prior_fns, pos, neighbor, **dynamic_kwargs):
    """Evaluates the prior potential for a given snapshot."""
    sum_priors = 0.
    if prior_fns is not None:
        for key in prior_fns:
            sum_priors += prior_fns[key](pos, neighbor=neighbor,
                                         **dynamic_kwargs)
    return sum_priors
