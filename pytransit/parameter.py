#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Model parameterization.

Parameters are named quantities with hard bounds and optional priors. Priors are
scipy.stats frozen distributions (or any object exposing `logpdf` and `rvs`); the
`normal_prior` and `uniform_prior` factories create the two supported distributions
using (mean, std) and (a, b) semantics.
"""

from typing import Sequence

from numpy import inf, where, atleast_2d, zeros, squeeze, array, ndarray, isfinite, stack, log, sqrt, pi
from numpy.random import uniform as _runiform
from scipy.stats import norm, uniform

__all__ = ['Parameter', 'ParameterBlock', 'ParameterSet',
           'normal_prior', 'uniform_prior', 'prior_kind', 'prior_limits', 'prior_moments',
           'describe_prior']


def normal_prior(mean: float, std: float):
    """Create a normal prior with a given mean and standard deviation."""
    return norm(mean, std)


def uniform_prior(a: float, b: float):
    """Create a uniform prior between a and b (note: scipy's uniform takes (loc, width))."""
    return uniform(a, b - a)


def prior_kind(prior) -> str | None:
    """Name of the distribution behind a prior ('norm', 'uniform', ...), or None."""
    if prior is None:
        return None
    return getattr(getattr(prior, 'dist', None), 'name', None)


def prior_limits(prior) -> tuple[float, float]:
    """The support of a prior as an (a, b) tuple."""
    a, b = prior.support()
    return float(a), float(b)


def prior_moments(prior) -> tuple[float, float]:
    """The mean and standard deviation of a prior."""
    return float(prior.mean()), float(prior.std())


def describe_prior(prior) -> str:
    """A short human-readable description of a prior."""
    kind = prior_kind(prior)
    if kind == 'norm':
        m, s = prior_moments(prior)
        return f'N(μ = {m:g}, σ = {s:g})'
    elif kind == 'uniform':
        a, b = prior_limits(prior)
        return f'U(a = {a:g}, b = {b:g})'
    elif prior is None:
        return '-'
    else:
        return str(prior)


class Parameter:
    def __init__(self, name, description='', unit='', prior=None, bounds=(-inf, inf), **kwargs):
        self.name: str = name
        self.description: str = description
        self.unit = unit
        self.prior = prior
        self._bounds: tuple[float, float] = bounds
        self.pid: int = -1

    def __str__(self):
        return f"{self.pid:3d} | {self.name:14s} | {str(self.unit):10s} |{describe_prior(self.prior):40} [{self.bounds[0]:8.2f} .. {self.bounds[1]:8.2f}]"

    def __repr__(self):
        return str(self)

    @property
    def bounds(self):
        return self._bounds

    def log_prior(self, v):
        if self.prior is None:
            return 0.0
        return self.prior.logpdf(v)

    def rvs(self, size):
        if self.prior is None:
            if isfinite(self.bounds[0]) and isfinite(self.bounds[1]):
                return _runiform(self.bounds[0], self.bounds[1], size)
            raise ValueError(f'Cannot sample parameter "{self.name}": it has no prior and unbounded limits.')
        return self.prior.rvs(size)


class ParameterBlock(list):
    def __init__(self, name, *args):
        super().__init__(*args)
        self.name = name
        self.start: int | None = None      # Set when the block is added to a ParameterSet
        self.stop: int | None = None
        self.slice: slice | None = None

    def add_parameter(self, name, description='', unit='', prior=None, bounds=(-inf, inf)):
        if name in self.names:
            raise ValueError(f'A parameter named "{name}" exists already.')
        super().append(Parameter(name, description, unit, prior, bounds))

    @property
    def names(self):
        return [p.name for p in self]


class ParameterSet(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.blocks: list[ParameterBlock] = []
        self.bounds = None
        self.frozen = False

    def add_parameter(self, name, description='', unit='', prior=None, bounds=(-inf, inf)):
        if name in self.names:
            raise ValueError(f'A parameter named "{name}" exists already.')
        if not self.frozen:
            super().append(Parameter(name, description, unit, prior, bounds))
        else:
            raise ValueError('Trying to add a parameter to a frozen ParameterSet')

    def add_parameter_block(self, block: ParameterBlock) -> ParameterBlock:
        if self.frozen:
            raise ValueError('Trying to add a parameter block to a frozen ParameterSet')
        if block.name in [b.name for b in self.blocks]:
            raise ValueError(f'A parameter block named "{block.name}" exists already.')
        for p in block:
            if p.name in self.names:
                raise ValueError(f'A parameter named "{p.name}" exists already.')

        s_pre = len(self)
        for p in block:
            super().append(p)
        s_post = len(self)
        block.start = s_pre
        block.stop = s_post
        block.slice = slice(s_pre, s_post)
        setattr(self, f'sl_{block.name.lower()}', block.slice)
        setattr(self, f'bl_{block.name.lower()}', self[block.slice])
        self.blocks.append(block)
        return block

    def add_block(self, name: str, parameters: Sequence[Parameter]) -> ParameterBlock:
        """Create a named parameter block from a sequence of parameters and add it to the set."""
        return self.add_parameter_block(ParameterBlock(name, parameters))

    def _update(self):
        if not self.frozen:
            self.bounds = array([p.bounds for p in self]).reshape(-1, 2)
            self.lbounds = self.bounds[:, 0]
            self.ubounds = self.bounds[:, 1]
            for i, p in enumerate(self):
                p.pid = i
        else:
            raise ValueError('Trying to update a frozen ParameterSet')

    def _refresh_prior_cache(self):
        """(Re)build the vectorized prior descriptors when the priors change.

        Normal and uniform priors are unpacked into plain arrays so `log_prior`
        and `log_prior_gradient` can run as a handful of numpy operations
        instead of one scipy frozen-distribution call per parameter (which
        dominates the cost of a single posterior evaluation). Staleness is
        detected by object identity against a held reference list, so priors
        replaced either through `set_prior` or by direct assignment to
        `parameter.prior` are picked up on the next call.
        """
        priors = [p.prior for p in self]
        cached = getattr(self, '_pc_priors', None)
        if cached is not None and len(cached) == len(priors) and \
                all(a is b for a, b in zip(cached, priors)):
            return
        norm_ids, norm_m, norm_s = [], [], []
        uni_ids, uni_a, uni_b = [], [], []
        other_ids = []
        for i, p in enumerate(self):
            kind = prior_kind(p.prior)
            if kind == 'norm':
                m, s = prior_moments(p.prior)
                norm_ids.append(i)
                norm_m.append(m)
                norm_s.append(s)
            elif kind == 'uniform':
                a, b = prior_limits(p.prior)
                uni_ids.append(i)
                uni_a.append(a)
                uni_b.append(b)
            elif p.prior is not None:
                other_ids.append(i)
        self._pc_priors = priors
        self._pc_nids = array(norm_ids, int)
        self._pc_nm = array(norm_m)
        self._pc_ns = array(norm_s)
        self._pc_nconst = -log(self._pc_ns * sqrt(2.0 * pi)).sum() if norm_ids else 0.0
        self._pc_uids = array(uni_ids, int)
        self._pc_ua = array(uni_a)
        self._pc_ub = array(uni_b)
        self._pc_uconst = -log(self._pc_ub - self._pc_ua).sum() if uni_ids else 0.0
        self._pc_oids = other_ids

    def log_prior(self, pv):
        pv = atleast_2d(pv)
        self._refresh_prior_cache()
        m = (pv > self.bounds[:, 0]).all(1) & (pv < self.bounds[:, 1]).all(1)
        lnp = zeros(pv.shape[0])
        if self._pc_nids.size:
            x = (pv[:, self._pc_nids] - self._pc_nm) / self._pc_ns
            lnp += self._pc_nconst - 0.5 * (x * x).sum(1)
        if self._pc_uids.size:
            x = pv[:, self._pc_uids]
            inside = ((x >= self._pc_ua) & (x <= self._pc_ub)).all(1)
            lnp = where(inside, lnp + self._pc_uconst, -inf)
        for i in self._pc_oids:
            lnp = lnp + self[i].log_prior(pv[:, i])
        return squeeze(where(m, lnp, -inf))

    def log_prior_gradient(self, pv) -> ndarray:
        """Gradient of the log prior density with respect to the parameter values.

        Supports normal and uniform priors and parameters without a prior (zero
        gradient inside the bounds); other prior distributions raise a
        NotImplementedError. The parameter values are assumed to be inside the
        bounds and the prior supports.
        """
        pv = atleast_2d(pv)
        self._refresh_prior_cache()
        if self._pc_oids:
            i = self._pc_oids[0]
            raise NotImplementedError(f'Log prior gradient is not implemented for the '
                                      f'"{prior_kind(self[i].prior)}" prior of parameter '
                                      f'"{self[i].name}".')
        g = zeros(pv.shape)
        if self._pc_nids.size:
            g[:, self._pc_nids] = -(pv[:, self._pc_nids] - self._pc_nm) / self._pc_ns ** 2
        return g[0] if g.shape[0] == 1 else g

    def freeze(self):
        self._update()
        self.frozen = True

    def thaw(self):
        self.frozen = False

    def find_pid(self, name):
        for p in self:
            if name == p.name:
                return p.pid
        raise KeyError('Could not find parameter {}'.format(name))

    def sample_from_prior(self, size=1):
        return stack([p.rvs(size) for p in self], 1)

    def check_pv(self, pv):
        for i, p in enumerate(self):
            lnp = p.log_prior(pv[i])
            b = self.bounds[i]
            is_finite = p.bounds[0] < pv[i] < p.bounds[1]
            print(f"|{p.pid:3d}| {'*' if not is_finite else ' '} {p.name:10} {b[0]:7.1f} < {pv[i]:10.2f} < {b[1]:<8.1f} {describe_prior(p.prior):40} log P = {lnp:6.2f}")

    @property
    def mean_pv(self):
        x0 = zeros(len(self))
        for i, p in enumerate(self):
            if p.prior is not None:
                x0[i] = float(p.prior.mean())
            elif isfinite(p.bounds[0]) and isfinite(p.bounds[1]):
                x0[i] = 0.5 * (p.bounds[0] + p.bounds[1])
            else:
                raise ValueError(f'Cannot compute a mean value for parameter "{p.name}": '
                                 'it has no prior and unbounded limits.')
        return x0

    def set_prior(self, name: str, prior, *nargs):
        pid = self.find_pid(name)
        if hasattr(prior, 'logpdf'):
            self[pid].prior = prior
        elif isinstance(prior, str):
            pname = prior.lower()
            if pname in ('u', 'up', 'uniform'):
                if len(nargs) != 2:
                    raise ValueError("The uniform prior requires two additional arguments: a and b.")
                self[pid].prior = uniform_prior(nargs[0], nargs[1])
            elif pname in ('n', 'np', 'normal'):
                if len(nargs) != 2:
                    raise ValueError("The normal prior requires two additional arguments: mean and std.")
                self[pid].prior = normal_prior(nargs[0], nargs[1])
            else:
                raise ValueError(f'Unknown prior "{prior}". Allowed string codes are (N)ormal and (U)niform.')
        else:
            raise ValueError('The prior must be either an object with a logpdf method or a string prior code.')

    def set_bounds(self, name: str, bounds: tuple[float, float]):
        pid = self.find_pid(name)
        self.thaw()
        self[pid]._bounds = bounds
        self._update()
        self.freeze()

    @property
    def names(self):
        return [p.name for p in self]

    @property
    def units(self):
        return [p.unit for p in self]

    @property
    def descriptions(self):
        return [p.description for p in self]

    @property
    def priors(self):
        return [p.prior for p in self]
