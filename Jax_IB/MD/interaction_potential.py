import jax.numpy as jnp
import numpy as np
from jax_md import space, smap, energy, minimize, quantity, simulate, partition


def harmonic_morse(dr, h=0.5, D0=5.0, alpha=5.0, r0=1.0, k=300.0, **kwargs):
    U = jnp.where(dr < r0, 
               h * k * (dr - r0)**2 - D0,
               D0 * (jnp.exp(-2. * alpha * (dr - r0)) - 2. * jnp.exp(-alpha * (dr - r0)))
               )
    return jnp.array(U, dtype=dr.dtype)

f32 = np.float32
f64 = np.float64
def harmonic_morse_pair(displacement_or_metric, species=None, h=0.5, D0=5.0, alpha=10.0, r0=1.0, k=50.0): 
    h = jnp.array(h, dtype=f32)
    D0 = jnp.array(D0, dtype=f32)
    alpha = jnp.array(alpha, dtype=f32)
    r0 = jnp.array(r0, dtype=f32)
    k = jnp.array(k, dtype=f32)
    return smap.pair(harmonic_morse,space.canonicalize_displacement_or_metric(displacement_or_metric),
                     species=species,h=h,D0=D0,alpha=alpha,r0=r0,k=k)

