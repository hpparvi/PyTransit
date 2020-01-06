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

"""Log posterior functions for transit modelling and parameter estimation.

A log posterior function (LPF) class creates a basis for Bayesian parameter estimation from transit light curves.
In PyTransit, LPFs are a bit more than what the name implies. An LPF stores the observations, model priors, etc.
It also contains methods for posterior optimisation and MCMC sampling.

"""

from .lpf import BaseLPF
from .baselines.legendrebaseline import LegendreBaseline