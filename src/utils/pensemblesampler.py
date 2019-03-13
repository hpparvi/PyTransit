#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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

import numpy as np

try:
    from emcee import EnsembleSampler
except ImportError:
    pass

class PEnsembleSampler(EnsembleSampler):

    def __init__(self, nwalkers, dim, lnpostfn, a=2.0, args=[], kwargs={},
                 postargs=None, threads=1, pool=None, live_dangerously=False,
                 parallel_lnpostfn=False, runtime_sortingfn=None):

        super().__init__(nwalkers, dim, lnpostfn, a, args, kwargs, postargs, threads, pool,
                         live_dangerously=live_dangerously, runtime_sortingfn=runtime_sortingfn)
        self.parallel_lnpostfn = parallel_lnpostfn

    def _get_lnprob(self, pos=None):
        """
        Calculate the vector of log-probability for the walkers.

        :param pos: (optional)
            The position vector in parameter space where the probability
            should be calculated. This defaults to the current position
            unless a different one is provided.

        This method returns:

        * ``lnprob`` - A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.

        * ``blob`` - The list of meta data returned by the ``lnpostfn`` at
          this position or ``None`` if nothing was returned.

        """
        if pos is None:
            p = self.pos
        else:
            p = pos

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite.")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN.")

        if self.parallel_lnpostfn:
            # Let the log posterior function handle the parallelisation if
            # `parallel_lnpostfn` has been enabled.
            results = self.lnprobfn(p)
        else:
            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method. Otherwise,
            # just use the built-in `map` function.
            if self.pool is not None:
                M = self.pool.map
            else:
                M = map

            # sort the tasks according to (user-defined) some runtime guess
            if self.runtime_sortingfn is not None:
                p, idx = self.runtime_sortingfn(p)

            # Run the log-probability calculations (optionally in parallel).
            results = list(M(self.lnprobfn, [p[i] for i in range(len(p))]))

        try:
            lnprob = np.array([float(l[0]) for l in results])
            blob = [l[1] for l in results]
        except (IndexError, TypeError):
            lnprob = np.array([float(l) for l in results])
            blob = None

        # sort it back according to the original order - get the same
        # chain irrespective of the runtime sorting fn
        if self.runtime_sortingfn is not None:
            orig_idx = np.argsort(idx)
            lnprob = lnprob[orig_idx]
            p = [p[i] for i in orig_idx]
            if blob is not None:
                blob = [blob[i] for i in orig_idx]

        # Check for lnprob returning NaN.
        if np.any(np.isnan(lnprob)):
            # Print some debugging stuff.
            print("NaN value of lnprob for parameters: ")
            for pars in p[np.isnan(lnprob)]:
                print(pars)

            # Finally raise exception.
            raise ValueError("lnprob returned NaN.")

        return lnprob, blob
