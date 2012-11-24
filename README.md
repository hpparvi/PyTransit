PyTransit
=========

Fast and easy-to-use tools for planet transit modelling using Python or Fortran.

Modules
-------
Transit models
  - Optimized Fortran implementation of the transit model by A. Gimenez (A&A 450, 1231--1237, 2006).

Utilities
  - Routines to calculate the projected planet-to-star distance for circular and eccentric orbits.
  - Routines to calculate the transit duration, etc.

Install
-------
Building and installing is simple using a fairly modern gfortran, use --user for a local installation without root rights.

    python setup.py config_fc --fcompiler=gnu95 --opt="-Ofast -ffast-math" --f90flags="-cpp -fopenmp -march=native" build
    python setup.py install [--user]
    
The code should compile without major problems with other compilers also, but only Intel fortran has been tested.

Author
------
  - Hannu Parviainen <hpparvi@gmail.com>, Instituto de Astrofísica de Canarias (IAC)

Publications using the code
----------------------------
  - Parviainen, Hannu, Hans Deeg, and Juan A. Belmonte. “Secondary Eclipses in the CoRoT Light Curves: A Homogeneous Search Based in Bayesian Model Selection.” Accepted to A&A (2012)
  - Rouan, D., H. Parviainen, C. Moutou, Magali Deleuil, M. Fridlund, A Ofir, M. Havel, et al. “Transiting Exoplanets from the CoRoT Space Mission.” Astronomy & Astrophysics 537 (January 9, 2012): A54.
  - Murgas, F., E. Pallé, A. Cabrera-Lavers, K. D. Colón, E. L. Martín, and H. Parviainen. “Narrow Band H α Photometry of the super-Earth GJ 1214b with GTC/OSIRIS Tunable Filters.” Astronomy & Astrophysics 544 (July 24, 2012): A41.
  - Tingley, Brandon, E. Palle, H. Parviainen, H. J. Deeg, M. R. Zapatero Osorio, A. Cabrera-Lavers, J. a. Belmonte, P. M. Rodriguez, F. Murgas, and I. Ribas. “Detection of Transit Timing Variations in Excess of One Hour in the Kepler Multi-planet Candidate System KOI 806 with the GTC.” Astronomy & Astrophysics 536 (December 12, 2011): L9.

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