

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jax_cfd.base import advection
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.base import pressure
from jax_cfd.base import time_stepping
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences
import tree_math
import particle_class 

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector

BCFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]
BCFn_new =  Callable[[GridVariableVector, float], GridVariableVector]
IBMFn =  Callable[[particle_class.All_Variables, float], GridVariableVector]
GradPFn = Callable[[GridVariable], GridArrayVector]

PosFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]

DragFn =  Callable[[particle_class.All_Variables], particle_class.All_Variables]




def explicit_Reserve_BC(
    ReserveBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Reserve_boundary(v, *args):
    return ReserveBC(v, *args)
   Reserve_bc_ = _wrap_term_as_vector(Reserve_boundary, name='Reserve_BC')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Reserve_bc(v):
       
       return Reserve_bc_(v,step_time)

   return _Reserve_bc

def explicit_update_BC(
    updateBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Update_boundary(v, *args):
    return updateBC(v, *args)
   Update_bc_ = _wrap_term_as_vector(Update_boundary, name='Update_BC')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Update_bc(v):
       
       return Update_bc_(v,step_time)

   return _Update_bc


def explicit_IBM_Force(
    cal_IBM_force: IBMFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def IBM_FORCE(v, *args):
    return cal_IBM_force(v, *args)
   IBM_FORCE_ = _wrap_term_as_vector(IBM_FORCE, name='IBM_FORCE')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _IBM_FORCE(v):
       
       return IBM_FORCE_(v,step_time)

   return _IBM_FORCE



def explicit_Update_position(
    cal_Update_Position: PosFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Update_Position(v, *args):
    return cal_Update_Position(v, *args)
   Update_Position_ = _wrap_term_as_vector(Update_Position, name='Update_Position')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Update_Position(v):
       
       return Update_Position_(v,step_time)

   return _Update_Position


def explicit_Calc_Drag(
    cal_Drag: DragFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Calculate_Drag(v, *args):
    return cal_Drag(v, *args)
   Calculate_Drag_ = _wrap_term_as_vector(Calculate_Drag, name='Calculate_Drag')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Calculate_Drag(v):
       
       return Calculate_Drag_(v,step_time)

   return _Calculate_Drag

def explicit_Pressure_Gradient(
    cal_Pressure_Grad: GradPFn,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Pressure_Grad(v):
    return cal_Pressure_Grad(v)
   Pressure_Grad_ = _wrap_term_as_vector(Pressure_Grad, name='Pressure_Grad')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Pressure_Grad(v):
       
       return Pressure_Grad_(v)

   return _Pressure_Grad
