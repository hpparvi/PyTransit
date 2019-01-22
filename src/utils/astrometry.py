from orbits import a_from_mp
from exotk.constants import msun, mjup

as_c = {'as':1., 'mas':1e3, 'muas':1e6}

def angular_signal(mass, period, distance, mstar=1., units='as'):
    """Angular semi-amplitude of the astrometric signal caused by a planet.
    
    Parameters
    ----------

      mass     : planet mass            [M_Jup]
      period   : orbital period         [d]
      distance : distance to the system [pc]
      mstar    : stellar mass           [M_Sun]
      units    : 'as', 'mas', or 'muas'

    Returns
    -------

      angular semi-amplitude of the astrometric signal by the planet in given units

    """
    return as_c[units] * (mass*mjup)/(mstar*msun) * a_from_mp(mstar, period) / distance
