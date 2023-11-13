

from typing import Callable, Optional
import jax.scipy.sparse.linalg
from jax_cfd.base import array_utils
from jax_cfd.base import fast_diagonalization
import jax.numpy as jnp
from jax_cfd.base import pressure
from jax_ib.base import grids
from jax_ib.base import boundaries
from jax_ib.base import finite_differences as fd
from jax_ib.base import particle_class

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions

def projection_and_update_pressure(
    All_variables: particle_class.All_Variables,
    solve: Callable = pressure.solve_fast_diag,
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


def solve_fast_diag(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None) -> GridArray:
  """Solve for pressure using the fast diagonalization approach."""
  del q0  # unused
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic velocity BC')
  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True, implementation=implementation)
  return grids.applied(pinv)(rhs)


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
