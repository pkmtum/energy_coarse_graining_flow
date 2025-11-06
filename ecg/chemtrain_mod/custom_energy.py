"""Addition to chemtrain's jax_md_mod.custom_energy.py at
https://github.com/tummfm/chemtrain. Custom definition of 
some potential energy functions."""
from jax import numpy as jnp

from functools import partial

from jax_md_mod import custom_quantity
from jax_md import util, space, energy

from jaxtyping import Array

f32 = util.f32
DisplacementOrMetricFn = space.DisplacementOrMetricFn


def harmonic_dihedral(displacement_or_metric: DisplacementOrMetricFn,
                      dihedral_idxs: Array,
                      eq_mean: Array = None,
                      eq_variance: Array = None,
                      kbt: tuple[float, Array] = None,
                      psi0: Array = None,
                      kpsi: Array = None,
                      ):
    """Harmonic Dihedral interaction (improper dihedral).

    The variance of the dihedral angle is used to determine the force constant.
    https://manual.gromacs.org/documentation/2019/reference-manual/functions/bonded-interactions.html

    Args:
        displacement_or_metric: Displacement function
        dihedral_idxs: Indices of particles (i, j, k, l) building the dihedrals
        eq_mean: Equilibrium dihedral angle in degrees
        eq_variance: Dihedral angle Variance
        kbt: kbT

    Returns:
        Harmonic dihedral potential energy function.
    """
    if psi0 is None:
        psi0 = eq_mean
    if kpsi is None:
        kbt = jnp.array(kbt, dtype=f32)
        kpsi = kbt / eq_variance

    harmonic_fn = partial(energy.simple_spring, length=psi0, epsilon=kpsi)

    def energy_fn(pos, **unused_kwargs):
        dihedral_angles = custom_quantity.dihedral_displacement(
            pos, displacement_or_metric, dihedral_idxs)

        return jnp.sum(harmonic_fn(dihedral_angles))

    return energy_fn
