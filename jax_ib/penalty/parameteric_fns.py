import jax.numpy as jnp
import numpy as np
def param_ellipse(geometry_param,theta):
      A = geometry_param[0]
      B = geometry_param[1]
      return A*B/jnp.sqrt((B*jnp.cos(theta))**2 + (A*jnp.sin(theta))**2) # Ellipse 

def param_rose(geometry_param,theta):
      A = geometry_param[0]
      B = geometry_param[1]
      return A*jnp.sin(B*theta) # Rose Shape
def param_snail(geometry_param,theta):
      A = geometry_param[0]
      B = geometry_param[1]
      return A*theta

def param_circle(geometry_param,theta):
      A = geometry_param[0]
      return A*jnp.ones_like(theta) 

def param_rot_ellipse(phi,geometry_param,theta):
      A = geometry_param[0]
      B = geometry_param[1]
      excc = jnp.sqrt(1-jnp.round((B/A)**2,6))
      return B/jnp.sqrt(1-(excc*jnp.cos(theta-phi))**2)  
      #return A*B/jnp.sqrt((B*jnp.cos(theta))**2 + (A*jnp.sin(theta))**2) # Ellipse 
