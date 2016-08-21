from numpy import exp, sqrt
from scipy.constants import c, k, h

def planck(T, wl):
    """Radiance as a function or black-body temperature and wavelength.

    Parameters
    ----------

      T   : Temperature  [K]
      wl  : Wavelength   [m]

    Returns
    -------

      B   : Radiance
    """
    return 2*h*c**2/wl**5 / (exp(h*c/(wl*k*T))-1)

def cn_fluxes(wl, Ttar, Tcon, wlref, cnref):
    """Relative target and contaminant star fluxes given a reference wavelength and a contamination factor.
    
    Calculates the relative target and contaminant star fluxes 
    for wavelengths `wl` given the target and comparison star
    temperatures, a reference wavelength, and a contamination 
    factor in the reference wavelength.

    Parameters
    ----------

      wl     : Wavelength [m]
      Ttar   : Target star effective temperature [K]
      Tcon   : Comparison star effective temperature [K]
      wlref  : Reference wavelength [m]
      cnref  : Contamination in the reference wavelength (0-1)

    Returns
    -------

      ftar  : Target flux
      fcon  : contaminant flux
    """
    ftar = (1-cnref) * planck(Ttar, wl) / planck(Ttar, wlref)
    fcon =    cnref  * planck(Tcon, wl) / planck(Tcon, wlref)
    return ftar, fcon


def contamination(wl, Ttar, Tcon, wlref, cnref):
    """Contamination given a reference wavelength and a contamination factor.
    
    Calculates the contamination factor for wavelengths `wl` 
    given the target and comparison star temperatures, a 
    reference wavelength, and a contamination factor in the 
    reference wavelength.


    Parameters
    ----------

      wl     : Wavelength [m]
      Ttar   : Target star effective temperature [K]
      Tcon   : Comparison star effective temperature [K]
      wlref  : Reference wavelength [m]
      cnref  : Contamination in the reference wavelength (0-1)

    Returns
    -------

      c : Contamination in the given wavelength(s)

    """
    ftar, fcon = cn_fluxes(wl, Ttar, Tcon, wlref, cnref)
    return fcon/(ftar+fcon)


def contaminated_k(k0, cn):
    """The apparent radius ratio for a contaminated transit.
    """
    return k0*sqrt(1-cn)
