from numpy import sqrt
from .instrument import Instrument
from .contamination import SMContamination, BBContamination
from .filter import ClearFilter, BoxcarFilter, TabulatedFilter


def true_radius_ratio(apparent_k, contamination):
    return apparent_k / sqrt(1 - contamination)

def apparent_radius_ratio(true_k, contamination):
    return true_k * sqrt(1 - contamination)

__all__ = "TLC Instrument ClearFilter BoxcarFilter TabulatedFilter true_radius_ratio apparent_radius_ratio".split()