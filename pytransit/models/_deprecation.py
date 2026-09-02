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

"""Deprecation helpers for the legacy transit model evaluation API.

The `evaluate_ps` and `evaluate_pv` methods are the pre-2.9 evaluation interface, superseded by
the single broadcasting `evaluate` method. They are deprecated in PyTransit 2.9 and will be
removed in PyTransit 3.0.
"""

from astropy.utils.decorators import deprecated

__all__ = ['deprecated_evaluation_method', 'DEPRECATED_IN', 'REMOVED_IN']

#: The version the legacy evaluation methods were deprecated in.
DEPRECATED_IN = '2.9'

#: The version the legacy evaluation methods will be removed in.
REMOVED_IN = '3.0'

_MESSAGE = (f'The {{func}} {{obj_type}} is deprecated since PyTransit {DEPRECATED_IN} and will be '
            f'removed in PyTransit {REMOVED_IN}.')


def deprecated_evaluation_method(alternative: str = 'evaluate'):
    """Mark a legacy transit model evaluation method as deprecated.

    Applies `astropy.utils.decorators.deprecated` with the versions and wording PyTransit uses for
    the legacy evaluation API. Besides raising an `AstropyDeprecationWarning` when the method is
    called, the decorator prepends a ``.. deprecated::`` directive to the method's docstring, so
    the deprecation is rendered in the API documentation without any extra markup.

    Parameters
    ----------
    alternative : str, optional
        Name of the method to use instead, quoted in the warning and in the documentation.

    Returns
    -------
    callable
        A decorator that marks the method deprecated.

    Notes
    -----
    A deprecated method must never be called from inside PyTransit itself, or a user calling the
    supported API would get the warning. Where `evaluate` shares an implementation with a
    deprecated method, the implementation lives in a private method that both call.
    """
    return deprecated(DEPRECATED_IN, message=_MESSAGE, obj_type='method', alternative=alternative)
