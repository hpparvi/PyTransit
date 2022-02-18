from typing import Union, Iterable, Optional

from numpy import arange
from numpy.random import seed, uniform, normal

from .. import QuadraticModel
from ..orbits import as_from_rhop, i_from_ba


def create_mock_light_curve(tobs: float = 5.0, texp: float = 60.0, passband: Union[str, Iterable[str]] = 'i',
                            epoch: Union[int, Iterable[int]] = 0, transit_pars: Optional[dict] = None,
                            noise: float = 3e-4, rseed: int = 0):
    ldc = dict(g=[0.62, 0.44], r=[0.44, 0.38], i=[0.33, 0.36], z=[0.25, 0.33],
               Kepler=[0.43, 0.38], TESS=[0.31, 0.35], CHEOPS=[0.44, 0.39])

    if isinstance(epoch, int):
        epochs = [epoch]
    else:
        epochs = epoch
    assert all([isinstance(ep, int) for ep in epochs]), "Epochs must be given as integers."

    if isinstance(passband, str):
        passbands = [passband]
    else:
        passbands = passband
    assert all([isinstance(pb, str) for pb in passbands]), "Passbands must be given as strings."
    assert all([pb in ldc for pb in passbands]), f"Passbands must be in {list(ldc.keys())}"

    if len(epochs) == 1 and len(passbands) > 1:
        epochs = len(passbands) * epochs
    elif len(passbands) == 1 and len(epochs) > 1:
        passbands = len(epochs) * passbands
    else:
        assert len(epochs) == len(passbands), "Number of epochs and passbands must be the same."

    n_transits = len(epochs)

    tp = dict(t0=0.0, period=2.4, ror=0.1, rho=1.5, b=0.5, ecc=0.0, omega=0.0)
    if transit_pars is not None:
        tp.update(transit_pars)
    tp['aor'] = as_from_rhop(tp['rho'], tp['period'])
    tp['inc'] = i_from_ba(tp['b'], tp['aor'])

    tm = QuadraticModel(interpolate=False)
    seed(rseed)

    times, fluxes = [], []
    nexp = tobs // (texp / 60 / 60)
    for iep, ep in enumerate(epochs):
        tc = tp['t0'] + tp['period'] * ep
        t = tc + (texp / 60 / 60 / 24 * (arange(nexp) + uniform())) - 0.5 * tobs / 24
        tm.set_data(t)
        f = tm.evaluate(tp['ror'], ldc[passbands[iep]], tp['t0'], tp['period'], tp ['aor'], tp['inc'], tp['ecc'], tp['omega'])
        f += normal(0.0, noise, size=f.size)
        times.append(t)
        fluxes.append(f)

    if n_transits == 1:
        return times[0], fluxes[0], tp
    else:
        return times, fluxes, tp
