from collections import namedtuple

from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools
import jax
from jax import grad
from jax import jit
from jax import random
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten

from jax_md import quantity
from jax_md import util
from jax_md import space
from jax_md import dataclasses
from jax_md import partition
from jax_md import smap
from jax_md import simulate

static_cast = util.static_cast


# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]

def brownian_cfd(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             kT: float,
             CFD_force,    
             gamma: float=0.1) -> Simulator:
  """Simulation of Brownian dynamics.

  Simulates Brownian dynamics which are synonymous with the overdamped
  regime of Langevin dynamics. However, in this case we don't need to take into
  account velocity information and the dynamics simplify. Consequently, when
  Brownian dynamics can be used they will be faster than Langevin. As in the
  case of Langevin dynamics our implementation follows Carlon et al. [#carlon]_

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.

  Returns:
    See above.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt, gamma = static_cast(dt, gamma)

  def init_fn(key, R, mass=f32(1)):
    state = simulate.BrownianState(R, mass, key)
    return simulate.canonicalize_mass(state)

  def apply_fn(all_variables, **kwargs):
    state = all_variables.MD_var    
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    R, mass, key = dataclasses.astuple(state)

    key, split = random.split(key)

    F = force_fn(R, **kwargs) + CFD_force(all_variables)
    xi = random.normal(split, R.shape, R.dtype)

    nu = f32(1) / (mass * gamma)

    dR = F * dt * nu + jnp.sqrt(f32(2) * _kT * dt * nu) * xi
    R = shift(R, dR, **kwargs)

    return simulate.BrownianState(R, mass, key)  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn
