import jax_ib.base as ib
import jax
import jax.numpy as jnp
import numpy as np

def arbitrary_obstacle(pressure_gradient,permeability):

  def forcing(v):

    #permeability = calc_perm(v)
    force_vector = (permeability, permeability)
    px = pressure_gradient
    return tuple(ib.grids.GridArray(pxn* jnp.ones_like(u.data)-f * u.data, u.offset, u.grid)
                 for pxn,f, u in zip(px,force_vector, v))        
    
  return forcing

def delta_approx_tanh(rf2,r2):
    inv_perm = 200000 
    width = 0.002
    r = jnp.sqrt(r2) 
    approx = inv_perm/width*r/jnp.cosh((rf2-r2)/width)**2
    
    return approx/np.max(approx)

def project_particle(grid,circle_center,Rtheta,delta_approx_fn):
    xc,yc = circle_center
    X, Y = grid.mesh(grid.cell_center)
    
    
    ntheta = Rtheta.size
    theta = jnp.array(np.linspace(0,2*jnp.pi,ntheta))
    theta_i = jnp.roll(theta,-1)

    Rtheta_i = jnp.array(jnp.roll(Rtheta,-1))


    theta_grid = jnp.arctan((Y-yc)/(X-xc))*jnp.heaviside(X-xc,1)*jnp.heaviside(Y-yc,0) # first quadrant

    theta_grid = theta_grid + (jnp.arctan((Y-yc)/(X-xc))+jnp.pi )*jnp.heaviside(xc-X,1) # second and third quadrant

    theta_grid = theta_grid + (jnp.arctan((Y-yc)/(X-xc))+ 2*jnp.pi)*jnp.heaviside(X-xc,0)*jnp.heaviside(yc-Y,0)  # second quadrant

    dtheta = 2*jnp.pi/(ntheta-1)
    #print(dtheta)

    theta_grid.flatten()


    #theta_0_grid = (theta_grid//dtheta).astype(int)
    #print(theta_0_grid)


    flattened_indx = (theta_grid//dtheta).astype(int)

    R0 = Rtheta[flattened_indx]
    R1 = Rtheta_i[flattened_indx]
    theta_0 = theta[flattened_indx]
    drdtheta = (Rtheta_i[flattened_indx]-Rtheta[flattened_indx])/(theta_i[flattened_indx]-theta_0)
    Rfinal = (R0 + drdtheta*(theta_grid - theta_0)).reshape(X.shape)
    #prefac = 90
    r2 = (Y-circle_center[1])**2 + (X-circle_center[0])**2
    distance_sq = (Rfinal-jnp.sqrt((Y-circle_center[1])**2 + (X-circle_center[0])**2))**2
    
    #delta_approx = jnp.exp(-prefac*distance_sq)
    #delta_approx = delta_approx_fn(distance_sq)
    delta_approx = delta_approx_tanh(Rfinal**2,r2)
    nx = (drdtheta*jnp.sin(theta_grid)+Rfinal*jnp.cos(theta_grid))
    ny = (Rfinal*jnp.sin(theta_grid)-drdtheta*jnp.cos(theta_grid))
    length = jnp.sqrt(nx**2 + ny**2)
    
    normal_v = (ib.grids.GridArray(nx/length*delta_approx,grid.cell_center,grid), ib.grids.GridArray(ny/length*delta_approx,grid.cell_center,grid))
    return normal_v,Rfinal


def calc_perm(grid,circle_center,Rtheta,smoothening_fn,Know):
    X,Y = grid.mesh(grid.cell_center)
    delta_approx = lambda r: delta_approx_fn(r,grid)
    normal_v,Rfinal = project_particle(grid,circle_center,Rtheta,delta_approx)

    del normal_v
    
    
    #width = 0.006#0.004*jnp.max(Rfinal)#0.00005

    #inv_perm = 20000#200000
    #G = (Rtheta[0]**2-((Y-circle_center[1])**2 + (X-circle_center[0])**2))/width
    G = (Rfinal**2-((Y-circle_center[1])**2 + (X-circle_center[0])**2))   ### ORIGINAL
   # return jnp.where(G>0,inv_perm*jnp.ones_like(G),jnp.zeros_like(G))
   # return smoothening_fn(((Y-circle_center[1])**2 + (X-circle_center[0])**2),Rtheta[0]**2,Know)
    return smoothening_fn(G,Know)
    #return inv_perm/2.0*(1.0 + jnp.tanh(G))
    #return inv_perm*jnp.heaviside(G,1.0)

def perm_vmap_multiple_particles(grid,particles,smoothening_fn,Know):
    
    def Vmap_calc_perm(grid,Grid_p,tree_arg):
      calc_r = tree_arg.shape 
      

      def foo(tree_arg):
        (particle_center,geometry_param,_,_) = tree_arg
        R_theta =  calc_r(geometry_param,Grid_p)
        return calc_perm(grid,particle_center,R_theta,smoothening_fn,Know)

      xs_flat, xs_tree = jax.tree_flatten(tree_arg)
      #print(xs_flat)  
      return jax.vmap(foo)(xs_flat)
    X,Y = grid.mesh()
    perm = jnp.zeros_like(X)
    
    if isinstance(particles, tuple):
        
      for particle in particles:
        Grid_p = particle.generate_grid()
        
        perm += jnp.sum(Vmap_calc_perm(grid,Grid_p,particle),axis=0)
      return perm

    else:
       Grid_p = particles.generate_grid()
       perm += jnp.sum(Vmap_calc_perm(grid,Grid_p,particles),axis=0)
       return perm
