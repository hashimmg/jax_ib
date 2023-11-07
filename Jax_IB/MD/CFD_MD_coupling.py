import jax.numpy as jnp
import jax

def surface_fn_jax(field,surface_coord):
    return jax.scipy.ndimage.map_coordinates(field, surface_coord, order=1)

def interpolate_pbc(field,list_p):
    list_p=jnp.moveaxis(list_p,0,-1)

    grid = field.grid
    offset = field.offset
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]

    xp=list_p[0]
    yp=list_p[1]
  
    surface_coord =(((xp)/dxEUL-offset[0]),((yp)/dyEUL-offset[1]))
    return surface_fn_jax(field.data,surface_coord)


def custom_force_fn_pbc(all_variables):#R, trajectory_cfd):
    trajectory_cfd = all_variables.velocity
    R = all_variables.MD_var.position
    u=interpolate_pbc(trajectory_cfd[0],R)
    v=interpolate_pbc(trajectory_cfd[1],R)
    return jnp.moveaxis(jnp.array([u,v]),0,-1)*1.0
