import jax_ib.base as ib
import jax
import jax.numpy as jnp

def arbitrary_obstacle(pressure_gradient,permeability):

  def forcing(v):

    #permeability = calc_perm(v)
    force_vector = (permeability, permeability)
    px = pressure_gradient
    return tuple(ib.grids.GridArray(pxn* jnp.ones_like(u.data)-f * u.data, u.offset, u.grid)
                 for pxn,f, u in zip(px,force_vector, v))        
    
  return forcing
