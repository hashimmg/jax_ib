import particle_class

from typing import Callable, Optional
import jax.numpy as jnp
from jax_cfd.base import pressure
from jax_cfd.base import grids
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd


def projection_and_update_pressure(
    All_variables: particle_class.All_Variables,
    solve: Callable = solve_fast_diag,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free."""
  v = All_variables.velocity
  old_pressure = All_variables.pressure
  particles = All_variables.particles
  Drag =  All_variables.Drag 
  Step_count = All_variables.Step_count
  MD_var = All_variables.MD_var
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = grids.GridVariable(q0, pressure_bc)

  qsol = solve(v, q0)
  q = grids.GridVariable(qsol, pressure_bc)
    
  New_pressure_Array =  grids.GridArray(qsol.data + old_pressure.data,qsol.offset,qsol.grid)  
  New_pressure = grids.GridVariable(New_pressure_Array,pressure_bc) 

  q_grad = fd.forward_difference(q)
  if boundaries.has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
    new_variable = particle_class.All_Variables(particles,v_projected,New_pressure,Drag,Step_count,MD_var)
  else:
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc).impose_bc()
        for u, q_g in zip(v, q_grad))
    new_variable = particle_class.All_Variables(particles,v_projected,New_pressure,Drag,Step_count,MD_var)
  return new_variable


def calc_P(
    v: GridVariableVector,
    solve: Callable = solve_fast_diag,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free."""
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = grids.GridVariable(q0, pressure_bc)

  q = solve(v, q0)
  q = grids.GridVariable(q, pressure_bc)

  return q
