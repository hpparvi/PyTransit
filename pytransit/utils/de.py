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

"""Differential evolution global optimiser.

Implements the differential evolution optimisation method by Storn & Price (Storn, R. & Price,
K. Journal of Global Optimization 11, 341 (1997)).
"""

from numba import njit
from numpy import asarray, zeros, zeros_like, tile, array, argmin, mod, ptp, int64, iinfo
from numpy.random import randint, rand, default_rng


def wrap(v, vmin, vmax):
    """Wrap values periodically into the range [vmin, vmax)."""
    w = vmax - vmin
    return vmin + mod(asarray(v) - vmin, w)


@njit
def evolve_vector(i, pop, f, c):
    """Create a trial vector for the population member `i` with the DE/rand/1/bin scheme.

    Parameters
    ----------
    i : int
        Index of the population member.
    pop : ndarray
        Population as a ``(npop, ndim)`` array.
    f : float
        Difference amplification factor.
    c : float
        Crossover probability.

    Returns
    -------
    ndarray
        The trial vector.
    """
    npop, ndim = pop.shape

    # --- Vector selection ---
    v1, v2, v3 = i, i, i
    while v1 == i:
        v1 = randint(npop)
    while (v2 == i) or (v2 == v1):
        v2 = randint(npop)
    while (v3 == i) or (v3 == v2) or (v3 == v1):
        v3 = randint(npop)

    # --- Mutation ---
    v = pop[v1] + f * (pop[v2] - pop[v3])

    # --- Cross over ---
    jf = randint(ndim)
    co = rand(ndim)
    for j in range(ndim):
        if co[j] > c and j != jf:
            v[j] = pop[i, j]
    return v


@njit
def _evolve_population(pop, pop2, f, c, rng):
    """Write a trial population created with the DE/rand/1/bin scheme into `pop2`.

    Parameters
    ----------
    pop : ndarray
        Current population as a ``(npop, ndim)`` array.
    pop2 : ndarray
        Array of the same shape as `pop` that receives the trial population.
    f : float
        Difference amplification factor.
    c : float
        Crossover probability.
    rng : Generator
        NumPy random number generator.
    """
    npop, ndim = pop.shape

    for i in range(npop):

        # --- Vector selection ---
        v1, v2, v3 = i, i, i
        while v1 == i:
            v1 = rng.integers(0, npop)
        while (v2 == i) or (v2 == v1):
            v2 = rng.integers(0, npop)
        while (v3 == i) or (v3 == v2) or (v3 == v1):
            v3 = rng.integers(0, npop)

        # --- Mutation ---
        v = pop[v1] + f * (pop[v2] - pop[v3])

        # --- Cross over ---
        co = rng.random(ndim)
        for j in range(ndim):
            if co[j] <= c:
                pop2[i, j] = v[j]
            else:
                pop2[i, j] = pop[i, j]

        # --- Forced crossing ---
        j = rng.integers(0, ndim)
        pop2[i, j] = v[j]


class DiffEvol(object):
    """Differential evolution optimiser.

    Implements the differential evolution optimisation method by Storn & Price (1997) with the
    DE/rand/1/bin scheme. By default, the amplification factor and the crossover probability are
    drawn anew for each generation from uniform distributions (dithering), which makes the
    optimiser robust without tuning.

    Parameters
    ----------
    fun : callable
        The function to minimise, called as ``fun(x, *args, **kwargs)``. With
        ``vectorize=True``, `x` is the whole population as an ``(npop, npar)`` array and the
        function must return an array of `npop` values.
    bounds : array_like
        Parameter bounds of the initial population as an ``(npar, 2)`` array. The bounds are not
        enforced after the initialisation.
    npop : int
        Population size. Values between 5 and 10 times the number of parameters work well.
    f : float, optional
        Difference amplification factor. Values of 0.5-0.8 are good in most cases. If None,
        the factor is drawn from `fbounds` for each generation.
    c : float, optional
        Crossover probability. Use 0.9 to test for fast convergence, and smaller values (~0.1)
        for a more elaborate search. If None, the probability is drawn from `cbounds` for each
        generation.
    seed : int or Generator, optional
        Seed for the optimiser's random number generator, or the generator itself. If None, the
        generator is seeded from NumPy's global random number generator, so that
        ``numpy.random.seed`` makes the optimisation reproducible.
    maximize : bool
        Maximise the function instead of minimising it.
    vectorize : bool
        Evaluate the function for the whole population in a single call.
    cbounds : tuple of two floats
        Range of the crossover probability when `c` is None.
    fbounds : tuple of two floats
        Range of the amplification factor when `f` is None.
    pool : object, optional
        Parallelisation pool with a `map` method, used to evaluate the trial population when the
        optimiser is not vectorised.
    min_ptp : float
        The optimisation stops when the peak-to-peak range of the population's function values
        falls below this value.
    args : list
        Additional positional arguments passed to `fun`.
    kwargs : dict
        Additional keyword arguments passed to `fun`.
    """

    def __init__(self, fun, bounds, npop, f=None, c=None, seed=None, maximize=False, vectorize=False, cbounds=(0.25, 1),
                 fbounds=(0.25, 0.75), pool=None, min_ptp=0.01, args=[], kwargs={}):
        if seed is None:
            seed = randint(0, iinfo(int64).max, dtype=int64)
        self._rng = default_rng(seed)

        self.minfun = _function_wrapper(fun, args, kwargs)
        self.bounds = asarray(bounds)
        self.n_pop = npop
        self.n_par = self.bounds.shape[0]
        self.bl = tile(self.bounds[:, 0], [npop, 1])
        self.bw = tile(self.bounds[:, 1] - self.bounds[:, 0], [npop, 1])
        self.m = -1 if maximize else 1
        self.vectorize = vectorize
        self.args = args
        self.pool = pool

        self.periodic = []
        self.min_ptp = min_ptp

        self.cmin = cbounds[0]
        self.cmax = cbounds[1]
        self.cbounds = cbounds
        self.fbounds = fbounds

        self.seed = seed
        self.f = f
        self.c = c

        self._population = asarray(self.bl + self._rng.random([self.n_pop, self.n_par]) * self.bw)
        self._fitness = zeros(npop)
        self._minidx = None

        self._trial_pop = zeros_like(self._population)
        self._trial_fit = zeros_like(self._fitness)
        self._prev_pop = zeros_like(self._population)
        self._prev_fit = zeros_like(self._fitness)

        if vectorize:
            self._eval = self._eval_vfun
        else:
            self._eval = self._eval_sfun

    @property
    def pool(self):
        """The parallelisation pool used to evaluate the population, or None.

        The pool needs only to provide a `map` method. Setting the pool also updates
        the mapping function used to evaluate the population, so the pool can safely be
        attached and detached between the optimisation runs. Note that the pool is used
        only if the optimiser is not vectorised.
        """
        return self._pool

    @pool.setter
    def pool(self, pool):
        self._pool = pool
        self.map = map if pool is None else pool.map

    @property
    def population(self):
        """The parameter vector population as an ``(npop, npar)`` array."""
        return self._population

    @property
    def minimum_value(self):
        """The best function value found. The value is negated when maximising."""
        return self._fitness[self._minidx]

    @property
    def minimum_location(self):
        """The parameter vector giving the best function value."""
        return self._population[self._minidx, :]

    @property
    def minimum_index(self):
        """Population index of the parameter vector giving the best function value."""
        return self._minidx

    def optimize(self, ngen):
        """Run the optimiser for a given number of generations.

        Parameters
        ----------
        ngen : int
            Maximum number of generations. The optimisation stops earlier if the population
            converges, see `min_ptp`.

        Returns
        -------
        tuple
            The best parameter vector and its function value (negated when maximising).
        """
        for _ in self(ngen):
            pass
        return self.minimum_location, self.minimum_value

    def __call__(self, ngen=1):
        """Return a generator that runs the optimiser for a given number of generations.

        The population is evaluated when the generator starts, after which each step evolves
        the population by one generation. The generator stops after `ngen` generations, or after
        the generation in which the population converges, see `min_ptp`.

        Parameters
        ----------
        ngen : int
            Maximum number of generations.

        Yields
        ------
        tuple
            The best parameter vector and its function value (negated when maximising) after each
            generation.
        """
        return self._eval(ngen)

    def _eval_sfun(self, ngen=1):
        """Run DE for a function that takes a single parameter vector and returns a single value."""
        popc, fitc = self._population, self._fitness
        popt, fitt = self._trial_pop, self._trial_fit

        fitc[:] = self.m * array(list(self.map(self.minfun, popc)))
        self._minidx = argmin(fitc)
        self._prev_pop[:, :] = popc
        self._prev_fit[:] = fitc

        for igen in range(ngen):
            f = self.f or self._rng.uniform(*self.fbounds)
            c = self.c or self._rng.uniform(*self.cbounds)

            _evolve_population(popc, popt, f, c, self._rng)
            fitt[:] = self.m * array(list(self.map(self.minfun, popt)))

            msk = fitt < fitc
            popc[msk, :] = popt[msk, :]
            fitc[msk] = fitt[msk]

            self._minidx = argmin(fitc)
            yield popc[self._minidx, :], fitc[self._minidx]

            if ptp(fitc) < self.min_ptp:
                break

    def _eval_vfun(self, ngen=1):
        """Run DE for a function that takes the whole population and returns a value for each parameter vector."""
        popc, fitc = self._population, self._fitness
        popt, fitt = self._trial_pop, self._trial_fit

        fitc[:] = self.m * self.minfun(self._population)
        self._minidx = argmin(fitc)
        self._prev_pop[:, :] = popc
        self._prev_fit[:] = fitc

        for igen in range(ngen):
            f = self.f or self._rng.uniform(*self.fbounds)
            c = self.c or self._rng.uniform(*self.cbounds)

            _evolve_population(popc, popt, f, c, self._rng)
            fitt[:] = self.m * self.minfun(popt)

            msk = fitt < fitc
            popc[msk, :] = popt[msk, :]
            fitc[msk] = fitt[msk]

            self._minidx = argmin(fitc)
            yield popc[self._minidx, :], fitc[self._minidx]

            if ptp(fitc) < self.min_ptp:
                break


class _function_wrapper(object):
    """Picklable wrapper that calls `f` with fixed additional arguments."""
    def __init__(self, f, args=[], kwargs={}):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)