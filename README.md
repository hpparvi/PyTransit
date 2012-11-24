PyTransit
=========

Fast and easy-to-use tools for planet transit modelling using Python or Fortran.

Transit models:
  Optimized Fortran implementation of the transit model by A. Gimenez (A&A 450, 1231--1237, 2006).


Author
  Hannu Parviainen <hpparvi@gmail.com>


Examples
--------
Calculate projected distance for a circular or eccentric orbit given time t:

    import numpy as np
    import pytransit.orbits_f as of

    t   = np.linspace(0.8,1.2,500)
    zc  = of.orbits.z_circular( t, 1, 4, 8, 0.48*np.pi, nthreads=0)                  
    zes = of.orbits.z_eccentric(t, 1, 4, 8, 0.48*np.pi, e=0.2, w=0.5, nthreads=0)    # Iteration
    zel = of.orbits.z_eccentric_ip(t, 1, 4, 8, 0.48*np.pi, e=0.2, w=0.5, nthreads=0) # Linear interpolation

Basic transit model usage:

    m = Gimenez() # Initialize the model, use quadratic limb darkening law and all available cores
    I = m(z,k,u)  # Evaluate the model for projected distance z, radius ratio k, and limb darkening coefficients u
      
Use linear interpolation:

    m = Gimenez(lerp=True) # Initialize the model
    I = m(z,k,u)           # Evaluate the model

Use linear interpolation, two different sets of z:

    m  = Gimenez(lerp=True)      # Initialize the model
    I1 = m(z1,k,u)               # Evaluate the model for z1, update the interpolation table
    I2 = m(z2,k,u, update=False) # Evaluate the model for z2, don't update the interpolation table